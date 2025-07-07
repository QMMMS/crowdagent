import os
import torch
import random
import copy
import yaml
import csv
import time
import pickle
import numpy as np
from datetime import datetime
from tqdm import trange, tqdm
from torch.optim import AdamW
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    AutoImageProcessor,
    AutoModelForImageClassification,
    WEIGHTS_NAME,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    MMBTConfig,
    MMBTForClassification,
)
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from src.slm_function.datasets_nll import ClassificationDataset, MyDataCollator
from transformers import AutoTokenizer, BertForSequenceClassification, RobertaForSequenceClassification
from src.slm_function.models.modeling_roberta import (
    RobertaForSequenceClassification_Modified,
)
from src.slm_function.utils.eval import evaluate_pll_single
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import src.slm_function.convnextv2 as convnextv2
import src.slm_function.chinese_roberta as chinese_roberta
import src.slm_function.robust_roberta as roberta
import logging
from src.slm_function.models.utils_mmimdb import ImageEncoder, CsvDataset, collate_fn, get_image_transforms, get_labels_from_label_num
from pprint import pprint
import src.utils.util as util

def get_args(task):
    with open("config/slm_config.yml", "r") as f:
        slm_config = yaml.safe_load(f)
    args = slm_config[task]

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args['GPUID']}"
    args["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    args["PROJECT_ROOT"] = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    args["CACHE_DIR"] = os.path.join(
        args["PROJECT_ROOT"], "src", "slm_function", "cached_models"
    )
    args["DATA_ROOT"] = os.path.join(args["PROJECT_ROOT"], "data")
    args["OUTPUT_DIR"] = os.path.join(args["PROJECT_ROOT"], "output", task)
    args["DATASET"] = task
    args["TASK"] = task
    args["TRAIN_BATCH_SIZE"] = args["TRAIN_BATCH"] * max(1, args["N_GPU"])
    args["IMAGE_DIR"] = os.path.join(args["PROJECT_ROOT"], "data", task, "images")
    return args


def set_logger(args, log_dir, name="slm_fliter"):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not any(
        isinstance(handler, logging.FileHandler) for handler in root_logger.handlers
    ):
        now_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        logging_fh = logging.FileHandler(
            os.path.join(log_dir, f"{name}_{now_time}.log")
        )
        logging_fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            "%m/%d/%Y %H:%M:%S",
        )
        logging_fh.setFormatter(formatter)
        root_logger.addHandler(logging_fh)

    if not root_logger.handlers:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if args["LOCAL_RANK"] in [-1, 0] else logging.WARN,
            handlers=[logging.StreamHandler()],
        )

    logger = logging.getLogger(__name__)
    for key in args:
        logger.info(f"{key}: {args[key]}")
    return logger


def set_seed(args):
    random.seed(args["SEED"])
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    if args["N_GPU"] > 0:
        torch.cuda.manual_seed_all(args["SEED"])


def calculate_chinese_roberta_outputs(args, logger, model_path, train_file_path):
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", cache_dir=args["CACHE_DIR"], local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.to(args["DEVICE"])
    train_dataset = chinese_roberta.ChineseRobertaDataset(
        csv_file_path=train_file_path,
        tokenizer=tokenizer,
        max_length=512,
        skip_header=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False
    )

    total_correct = 0
    total_samples = 0

    length = len(train_dataloader.dataset)
    outputs_all = torch.zeros((length, args["NUM_LABELS"])).to(args["DEVICE"])
    targets_all = torch.zeros((length), dtype=torch.long).to(args["DEVICE"])
    pred_idx_all = torch.zeros((length), dtype=torch.long).to(args["DEVICE"])
    raw_idx_all = [''] * length
    cls_embedding_all = torch.zeros((length, 768)).to(args["DEVICE"])

    idx = 0

    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            inputs = {k: v.to(args["DEVICE"]) for k, v in batch['inputs'].items()}
            labels = batch['labels'].to(args["DEVICE"])
            
            outputs = model(**inputs, output_hidden_states=True)
            predictions = outputs.logits.argmax(-1)
            cls_embedding = outputs.hidden_states[-1][:, 0, :]

            batch_size = labels.size(0)
            cls_embedding_all[idx:idx+batch_size, :] = cls_embedding
            targets_all[idx:idx+batch_size] = labels
            outputs_all[idx:idx+batch_size, :] = outputs.logits
            pred_idx_all[idx:idx+batch_size] = predictions
            for i in range(batch_size):
                raw_idx_all[idx + i] = batch["raw_index"][i]
            idx += batch_size

            
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        accuracy = total_correct / total_samples

        logger.info(f"trained model on train set Accuracy: {accuracy:.4f}")

    return outputs_all, targets_all, pred_idx_all, cls_embedding_all, raw_idx_all


def calculate_roberta_outputs(args, logger, model_path, train_file_path):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", cache_dir=args["CACHE_DIR"], local_files_only=True)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.to(args["DEVICE"])
    
    train_dataset = roberta.RobertaDataset(
        csv_file_path=train_file_path,
        tokenizer=tokenizer,
        max_length=512,
        skip_header=True,
        confidence_threshold=0,
        mode=args["TASK"],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False
    )

    total_correct = 0
    total_samples = 0

    length = len(train_dataloader.dataset)
    outputs_all = torch.zeros((length, args["NUM_LABELS"])).to(args["DEVICE"])
    targets_all = torch.zeros((length), dtype=torch.long).to(args["DEVICE"])
    pred_idx_all = torch.zeros((length), dtype=torch.long).to(args["DEVICE"])
    raw_idx_all = [''] * length
    cls_embedding_all = torch.zeros((length, 768)).to(args["DEVICE"])

    idx = 0

    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            inputs = {k: v.to(args["DEVICE"]) for k, v in batch['inputs'].items()}
            labels = batch['labels'].to(args["DEVICE"])
            
            outputs = model(**inputs, output_hidden_states=True)
            predictions = outputs.logits.argmax(-1)
            cls_embedding = outputs.hidden_states[-1][:, 0, :]

            batch_size = labels.size(0)
            cls_embedding_all[idx:idx+batch_size, :] = cls_embedding
            targets_all[idx:idx+batch_size] = labels
            outputs_all[idx:idx+batch_size, :] = outputs.logits
            pred_idx_all[idx:idx+batch_size] = predictions
            for i in range(batch_size):
                raw_idx_all[idx + i] = batch["raw_index"][i]
            idx += batch_size

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        logger.info(f"trained model on train set Accuracy: {accuracy:.4f}")

    return outputs_all, targets_all, pred_idx_all, cls_embedding_all, raw_idx_all


def calculate_mmbt_outputs(args, logger, model_path, train_file_path):
    labels = get_labels_from_label_num(args["NUM_LABELS"])
    num_labels = len(labels)
    transformer_config = AutoConfig.from_pretrained(
        "google-bert/bert-base-uncased",
        cache_dir=args["CACHE_DIR"],
        local_files_only=True,
    )
    config = MMBTConfig(transformer_config, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-uncased",
        do_lower_case=False,
        cache_dir=args["CACHE_DIR"],
        local_files_only=True,
    )
    transformer = AutoModel.from_pretrained(
        "google-bert/bert-base-uncased", config=transformer_config, cache_dir=args["CACHE_DIR"],
        local_files_only=True,
    )
    img_encoder = ImageEncoder(args)
    model = MMBTForClassification(config, transformer, img_encoder)
    model.load_state_dict(torch.load(os.path.join(model_path, WEIGHTS_NAME)))
    model.to(args["DEVICE"])
    model.eval()

    all_dataset = CsvDataset(
        csv_data_path=train_file_path,
        tokenizer=tokenizer,
        transforms=get_image_transforms(),
        image_base_dir=args["IMAGE_DIR"],
        max_seq_length=args["max_seq_length"] - args["num_image_embeds"] - 2,
        num_labels=args["NUM_LABELS"],
        mode=args["TASK"]
    )
    sampler = RandomSampler(all_dataset)
    all_dataloader = DataLoader(
        all_dataset,
        sampler=sampler,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
    )

    length = len(all_dataloader.dataset)
    outputs_all = torch.zeros((length, num_labels)).to(args["DEVICE"])
    targets_all = torch.zeros((length), dtype=torch.long).to(args["DEVICE"])
    pred_idx_all = torch.zeros((length), dtype=torch.long).to(args["DEVICE"])
    raw_idx_all = [''] * length
    cls_embedding_all = torch.zeros((length, 768)).to(args["DEVICE"])
    
    idx = 0

    with torch.no_grad():
        for batch in tqdm(all_dataloader):
            labels = batch[5].to(args["DEVICE"])
            inputs = {
                "input_ids": batch[0].to(args["DEVICE"]),
                "input_modal": batch[2].to(args["DEVICE"]),
                "attention_mask": batch[1].to(args["DEVICE"]),
                "modal_start_tokens": batch[3].to(args["DEVICE"]),
                "modal_end_tokens": batch[4].to(args["DEVICE"]),
                "return_dict": True
            }
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = logits.argmax(-1)
            try:
                cls_embedding = outputs.hidden_states[-1][:, 0, :]
            except:
                raise ValueError("""No hidden states found, try add output_hidden_states=True in MMBTForClassification:
outputs = self.mmbt(
input_modal=input_modal,
input_ids=input_ids,
modal_start_tokens=modal_start_tokens,
modal_end_tokens=modal_end_tokens,
attention_mask=attention_mask,
token_type_ids=token_type_ids,
modal_token_type_ids=modal_token_type_ids,
position_ids=position_ids,
modal_position_ids=modal_position_ids,
head_mask=head_mask,
inputs_embeds=inputs_embeds,
output_hidden_states=True,  # add this line
return_dict=return_dict,
)""")
            batch_size = labels.size(0)
            cls_embedding_all[idx:idx+batch_size, :] = cls_embedding
            targets_all[idx:idx+batch_size] = labels.argmax(-1)
            outputs_all[idx:idx+batch_size, :] = outputs.logits
            pred_idx_all[idx:idx+batch_size] = predictions
            for i in range(batch_size):
                raw_idx_all[idx + i] = batch[6][i]
            idx += batch_size

    accuracy = 0
    for i in range(len(targets_all)):
        if pred_idx_all[i] == targets_all[i]:
            accuracy += 1
    accuracy = accuracy / len(targets_all)
    logger.info(f"trained model on train set Accuracy: {accuracy:.4f}")

    return outputs_all, targets_all, pred_idx_all, cls_embedding_all, raw_idx_all


def calculate_roberta_embeddings(args, logger, model_path, train_file_path):
    model_s1 = RobertaForSequenceClassification_Modified.from_pretrained(model_path)
    model_s1.to(args["DEVICE"])
    model_s1.eval()
    logger.info("Model loaded")
    tokenizer = RobertaTokenizer.from_pretrained(
        args["TOKENIZER_NAME"],
        do_lower_case=args["DO_LOWER_CASE"],
        cache_dir=args["CACHE_DIR"] if args["CACHE_DIR"] else None,
        local_files_only=True,
    )
    train_dataset_aug = ClassificationDataset(
        args["DATA_ROOT"],
        tokenizer,
        args["DATASET"],
        train_file_path,
        args["MAX_SEQ_LENGTH"],
        args["OVERWRITE_CACHE"],
        mode="train_chat",
        num_labels=args["NUM_LABELS"],
        label_reduce=args["LABEL_REDUCE"],
        label_source=args["LABEL_SOURCE"],
    )
    data_collator = MyDataCollator()
    train_sampler = (
        RandomSampler(train_dataset_aug)
        if args["LOCAL_RANK"] == -1
        else DistributedSampler(train_dataset_aug)
    )
    train_dataloader = DataLoader(
        train_dataset_aug,
        sampler=train_sampler,
        batch_size=args["TRAIN_BATCH_SIZE"],
        collate_fn=data_collator,
    )
    length = len(train_dataloader.dataset)
    outputs_all = torch.zeros((length, args["NUM_LABELS"])).to(args["DEVICE"])
    targets_all = torch.zeros((length), dtype=torch.long).to(args["DEVICE"])
    embeddings_all = torch.zeros((length, args["EMBEDDING_DIM"]))

    for step, inputs in enumerate(tqdm(train_dataloader, desc="Processing")):
        inputs = {k: v.to(args["DEVICE"]) for k, v in inputs.items()}
        with torch.no_grad():
            labels = inputs["labels"]
            index = inputs["index"]
            valid_mask = labels >= 0
            del inputs["index"]
            del inputs["labels"]
            myinputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "is_feat": True,
            }
            embeddings, logits1 = model_s1(**myinputs)
            outputs_all[index] = logits1
            targets_all[index] = labels
            embeddings_all[index] = embeddings.detach().cpu()

    pred_idx_all = outputs_all.max(dim=-1)[1]
    return outputs_all, targets_all, pred_idx_all, embeddings_all


def choose_clean_set(args, logger, combined_list, high_confidence_indices, clean_rate=0.2):
    valid_idx_all = [sample[0] for sample in combined_list if sample[2] >= 0]
    selected_idx_all = [sample[0] for sample in combined_list if sample[0] in valid_idx_all and sample[0] not in high_confidence_indices]
    selected_samples = [sample for sample in combined_list if sample[0] in selected_idx_all]
    chosen_idx_sel = []
    for j in range(args["NUM_LABELS"]):
        selected_samples_j = [sample for sample in selected_samples if sample[1] == j]
        selected_samples_j = sorted(selected_samples_j, key=lambda x: x[3][j], reverse=True)
        selected_samples_j = selected_samples_j[:int(len(selected_samples_j) * clean_rate)]

        for sample in selected_samples_j[:5]:
            logger.info(f"idx: {sample[0]}, pred_idx: {sample[1]}, target_idx: {sample[2]}, confidence: {sample[3][j]}")

        chosen_idx_sel.extend([sample[0] for sample in selected_samples_j])

    logger.info(f"chosen_idx_sel: {chosen_idx_sel}")
    return chosen_idx_sel


def validate_clean_set(args, logger, pred_idx_all, chosen_idx_sel, ground_truth_file_path):
    chosen_list_all = chosen_idx_sel.numpy().tolist()
    pred_labels = pred_idx_all.to('cpu').numpy().tolist()

    with open(ground_truth_file_path, "r") as f:
        reader = csv.reader(f)
        ground_truth_data = list(reader)
    ground_truth_label_list = [row[0] for row in ground_truth_data]
    if len(ground_truth_label_list) != len(pred_labels):
        raise ValueError("ground_truth_label_list 与 pred_labels 长度不一致")
    
    chosen_gound_truth_label = [ground_truth_label_list[i] for i in chosen_list_all]
    chosen_pred_label = [pred_labels[i] for i in chosen_list_all]
    total_num = len(chosen_list_all)
    correct_num = sum([1 for i, j in zip(chosen_gound_truth_label, chosen_pred_label) if i == str(j)])
    accuracy = correct_num / total_num
    logger.info(f"accuracy of clean set: {accuracy:.4f}")
    return accuracy


def calculate_convnextv2_outputs(args, logger, model_path, train_file_path):
    processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-22k-384", cache_dir=args["CACHE_DIR"])
    model = AutoModelForImageClassification.from_pretrained(
        model_path, 
        cache_dir=args["CACHE_DIR"],
    )
    model.eval()
    model.to(args["DEVICE"])
    train_dataset = convnextv2.ConvNextV2Dataset(train_file_path, processor, args["IMAGE_DIR"])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,  
        shuffle=False,
    )
    total_correct = 0
    total_samples = 0

    length = len(train_dataloader.dataset)
    outputs_all = torch.zeros((length, args["NUM_LABELS"])).to(args["DEVICE"])
    targets_all = torch.zeros((length), dtype=torch.long).to(args["DEVICE"])
    pred_idx_all = torch.zeros((length), dtype=torch.long).to(args["DEVICE"])
    embeddings_all = torch.zeros((length, 768*12*12)).to(args["DEVICE"])
    raw_idx_all = [''] * length
    idx = 0

    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            inputs = {k: v.to(args["DEVICE"]) for k, v in batch['inputs'].items()}
            labels = batch['labels'].to(args["DEVICE"])
            
            outputs = model(**inputs, output_hidden_states=True)
            predictions = outputs.logits.argmax(-1)
            last_hidden_state = outputs.hidden_states[-1]
            last_hidden_state = last_hidden_state.view(last_hidden_state.size(0), -1)

            batch_size = labels.size(0)
            targets_all[idx:idx+batch_size] = labels
            outputs_all[idx:idx+batch_size, :] = outputs.logits
            pred_idx_all[idx:idx+batch_size] = predictions
            embeddings_all[idx:idx+batch_size, :] = last_hidden_state
            for i in range(batch_size):
                raw_idx_all[idx + i] = batch['raw_index'][i]
            idx += batch_size

            
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        accuracy = total_correct / total_samples

        logger.info(f"Accuracy: {accuracy:.4f}")
    
    return outputs_all, targets_all, pred_idx_all, embeddings_all, raw_idx_all


def get_all_predictions(task, model_path, confidence_csv_path, log_dir, out_path, type="chinese_roberta"):
    args = get_args(task)
    logger = set_logger(args, log_dir, name="slm_fliter")
    set_seed(args)
    if type == "chinese_roberta":
        outputs_all, targets_all, pred_idx_all, embeddings_all, raw_idx_all = calculate_chinese_roberta_outputs(args, logger, model_path, confidence_csv_path)
    elif type == "convnextv2":
        outputs_all, targets_all, pred_idx_all, embeddings_all, raw_idx_all = calculate_convnextv2_outputs(args, logger, model_path, confidence_csv_path)
    elif type == "roberta":
        outputs_all, targets_all, pred_idx_all, embeddings_all, raw_idx_all = calculate_roberta_outputs(args, logger, model_path, confidence_csv_path)
    elif type == "mmbt":
        outputs_all, targets_all, pred_idx_all, embeddings_all, raw_idx_all = calculate_mmbt_outputs(args, logger, model_path, confidence_csv_path)
    else:
        raise ValueError(f"Invalid model type: {type}")

    outputs_all = outputs_all.cpu().numpy().tolist()
    targets_all = targets_all.cpu().numpy().tolist()
    pred_idx_all = pred_idx_all.cpu().numpy().tolist()
    embeddings_all = embeddings_all.cpu().numpy().tolist()

    with open(confidence_csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        sample_dict = {row[0]: row for row in reader}
    with open(out_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header[:4])
        for i in range(len(raw_idx_all)):
            idx = raw_idx_all[i]
            sample = sample_dict[idx]
            pred_label = pred_idx_all[i]
            writer.writerow([idx, pred_label] + sample[2:4])


def fliter(task, model_path, confidence_csv_path, log_dir, confidence_csv_out_path, other_confidence_csv_out_path, most_representative_idx_output_file_path="", type="chinese_roberta", confidence_threshold=0.9, clean_rate=0.2, data_embeddings_path=""):
    args = get_args(task)
    logger = set_logger(args, log_dir, name="slm_fliter")
    set_seed(args)
    if type == "chinese_roberta":
        outputs_all, targets_all, pred_idx_all, embeddings_all, raw_idx_all = calculate_chinese_roberta_outputs(args, logger, model_path, confidence_csv_path)
    elif type == "convnextv2":
        outputs_all, targets_all, pred_idx_all, embeddings_all, raw_idx_all = calculate_convnextv2_outputs(args, logger, model_path, confidence_csv_path)
    elif type == "roberta":
        outputs_all, targets_all, pred_idx_all, embeddings_all, raw_idx_all = calculate_roberta_outputs(args, logger, model_path, confidence_csv_path)
    elif type == "mmbt":
        outputs_all, targets_all, pred_idx_all, embeddings_all, raw_idx_all = calculate_mmbt_outputs(args, logger, model_path, confidence_csv_path)
    else:
        raise ValueError(f"Invalid model type: {type}")
    high_confidence_indices = util.get_high_confidence_indices(
        confidence_csv_path, 
        confidence_threshold=confidence_threshold
    )
    logger.info(f"high_confidence_indices: {high_confidence_indices}, len: {len(high_confidence_indices)}")
    logger.info(f"raw_idx_all[:5]: {raw_idx_all[:5]}")
    logger.info(f"pred_idx_all[:5]: {pred_idx_all[:5]}")
    logger.info(f"targets_all[:5]: {targets_all[:5]}")

    if data_embeddings_path:
        torch.save({
            'raw_idx_all': raw_idx_all,
            'embeddings_all': embeddings_all
        }, data_embeddings_path)

    outputs_all = outputs_all.cpu().numpy().tolist()
    targets_all = targets_all.cpu().numpy().tolist()
    pred_idx_all = pred_idx_all.cpu().numpy().tolist()
    embeddings_all = embeddings_all.cpu().numpy().tolist()

    combined_list = []
    for i in range(len(raw_idx_all)):
        combined_list.append((raw_idx_all[i], pred_idx_all[i], targets_all[i], outputs_all[i], embeddings_all[i]))
    chosen_idx_sel = choose_clean_set(args, logger, combined_list, high_confidence_indices, clean_rate=clean_rate)

    with open(confidence_csv_path, "r") as f:
        reader = csv.reader(f)
        head = next(reader)
        confidence_data = [row for row in reader if row[0] in chosen_idx_sel]
        logger.info(f"len(confidence_data): {len(confidence_data)}")

    with open(confidence_csv_path, "r") as f:
        reader = csv.reader(f)
        head = next(reader)
        other_confidence_data = [row for row in reader if row[0] not in chosen_idx_sel and row[0] not in high_confidence_indices]
        logger.info(f"len(other_confidence_data): {len(other_confidence_data)}")

    with open(confidence_csv_out_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(head[:4])
        for row in confidence_data:
            raw_idx = row[0]
            slm_label = -1
            for sample in combined_list:
                if sample[0] == raw_idx:
                    slm_label = sample[1]
                    break
            writer.writerow([row[0], slm_label, row[2], row[3]])

    with open(other_confidence_csv_out_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(head[:4])
        for row in other_confidence_data:
            raw_idx = row[0]
            slm_label = -1
            for sample in combined_list:
                if sample[0] == raw_idx:
                    slm_label = sample[1]
                    break
            writer.writerow([row[0], slm_label, row[2], row[3]])

    high_conf_list = chosen_idx_sel + high_confidence_indices
    most_representative_idx = get_most_representative_idx(args, logger, confidence_csv_path, combined_list, high_conf_list, most_representative_idx_output_file_path)


def get_most_representative_idx(args, logger, confidence_csv_path, combined_list, high_conf_list, most_representative_idx_output_file_path=""):
    most_representative_idx = []
    for j in range(args["NUM_LABELS"]):
        index_j_sel = []
        embeddings_j = []
        for sample in combined_list:
            if sample[2] == j and sample[0] in high_conf_list:
                index_j_sel.append(sample[0])
                embeddings_j.append(sample[4])
        logger.info(f"index_j_sel: {index_j_sel}")

        embeddings_j = torch.tensor(embeddings_j)
        num_clusters = max(args["SELECT_DEMO_NUM"] // args["NUM_LABELS"], 1)
        if embeddings_j.shape[0] < num_clusters:
            num_clusters = embeddings_j.shape[0]
            logger.warning(f"embeddings_j.shape[0] < num_clusters, num_clusters is set to {num_clusters}")
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings_j)
        kmeans_labels = kmeans.labels_
        idx_all_representative = []

        for k in range(num_clusters):
            vectors_in_cluster = embeddings_j[kmeans_labels == k]
            idx_in_cluster = [index_j_sel[i] for i in range(len(index_j_sel)) if kmeans_labels[i] == k]
            centroid = vectors_in_cluster.mean(dim=0)
            distances_to_centroid = torch.norm(vectors_in_cluster - centroid, dim=1)
            index_of_representative = torch.argmin(distances_to_centroid)
            idx_all_representative.append(idx_in_cluster[index_of_representative.item()])

        most_representative_idx.extend(idx_all_representative)

    logger.info(f"most_representative_idx: {most_representative_idx}")

    with open(confidence_csv_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [row for row in reader]

    if most_representative_idx_output_file_path:
        with open(most_representative_idx_output_file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header[:4])
            for representative in most_representative_idx:
                for row in data:
                    if row[0] == representative:
                        writer.writerow(row)
                        break

    return most_representative_idx


def slm_annotation(logger, former_entropy_path, pred_idx_all, chosen_idx_sel, output_file_path, output_entropy_log_file_path, entropy_threshold=0.1):
    with open(former_entropy_path, 'r') as f:
        reader = csv.reader(f)
        entropy_log_header = next(reader)
        entropy_log_data = [row for row in reader]

    if len(entropy_log_data) != len(pred_idx_all):
        raise ValueError("entropy_log_data 与 pred_idx_all 长度不一致")
    
    clean_slm_weight = 200
    other_slm_weight = 10
    weights = []
    for i in range(4, len(entropy_log_header)):
        weights.append(int(entropy_log_header[i].split("(x")[1].split(")")[0]))
    weights.append(clean_slm_weight)
    weights.append(other_slm_weight)

    count_skip = 0
    count_clean = 0
    count_other = 0

    with open(output_file_path, 'w') as f, \
         open(output_entropy_log_file_path, 'w') as f_entropy:
        output_writer = csv.writer(f)
        entropy_log_writer = csv.writer(f_entropy)
        entropy_log_writer.writerow(entropy_log_header+[f"slm_clean(x{clean_slm_weight})", f"slm(x{other_slm_weight})"])

        for i in range(len(entropy_log_data)):
            if float(entropy_log_data[i][3]) < entropy_threshold:
                output_writer.writerow(entropy_log_data[i][:3])
                entropy_log_writer.writerow(entropy_log_data[i]+['#', '#'])
                count_skip += 1
            elif i in chosen_idx_sel:
                output_writer.writerow(entropy_log_data[i][:3])
                all_labels = entropy_log_data[i][4:] + [pred_idx_all[i]] + ["#"]
                entropy = util.calculate_entropy_with_weights(all_labels, weights)
                entropy_log_writer.writerow(entropy_log_data[i][:3] + [entropy] + all_labels)
                count_clean += 1
            else:
                all_labels = entropy_log_data[i][4:] + ["#"] + [pred_idx_all[i]]
                most_common_label = util.get_most_common_label(all_labels, weights)
                entropy = util.calculate_entropy_with_weights(all_labels, weights)
                output_writer.writerow([most_common_label] + entropy_log_data[i][1:3])
                entropy_log_writer.writerow([most_common_label] + entropy_log_data[i][1:3] + [entropy] + all_labels)
                count_other += 1

    logger.info(f"count_skip: {count_skip}")
    logger.info(f"count_clean: {count_clean}")
    logger.info(f"count_other: {count_other}")
