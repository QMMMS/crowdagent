import os
import yaml
import torch
import random
import logging
from datetime import datetime
from src.slm_function.models.utils_mmimdb import ImageEncoder, CsvDataset, collate_fn, get_image_transforms, get_labels_from_label_num
import transformers
from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    MMBTConfig,
    MMBTForClassification,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import numpy as np
import csv
from src.utils import util

def get_args(task):
    with open("config/slm_config.yml", "r") as f:
        slm_config = yaml.safe_load(f)
    args = slm_config[task]
    args["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    args["PROJECT_ROOT"] = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    args["CACHE_DIR"] = os.path.join(
        args["PROJECT_ROOT"], "src", "slm_function", "cached_models"
    )
    args["IMAGE_DIR"] = os.path.join(args["PROJECT_ROOT"], "data", task, "images")
    args["DATA_ROOT"] = os.path.join(args["PROJECT_ROOT"], "data")
    args["OUTPUT_DIR"] = os.path.join(args["PROJECT_ROOT"], "output", task)
    return args


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_logger(args, log_dir, name="mmbt"):
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


def evaluate_model(args, model, val_dataloader, batch_idx, epoch, logger):
    model.eval()
    targets_all = []
    total_pred_list = []
    raw_index_list = []
    chosen_list = []
    loss_all = torch.zeros((len(val_dataloader.dataset))).to(args["DEVICE"])

    idx = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            labels = batch[5].to(args["DEVICE"])

            inputs = {
                "input_ids": batch[0].to(args["DEVICE"]),
                "input_modal": batch[2].to(args["DEVICE"]),
                "attention_mask": batch[1].to(args["DEVICE"]),
                "modal_start_tokens": batch[3].to(args["DEVICE"]),
                "modal_end_tokens": batch[4].to(args["DEVICE"]),
                "return_dict": False
            }
            
            outputs = model(**inputs)
            logits = outputs[0]
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).detach().cpu().numpy()
            out_label_ids = torch.argmax(labels, dim=1).detach().cpu().numpy()

            batch_size = labels.size(0)
            loss_all[idx:idx+batch_size] = loss.detach()
            raw_index_list.extend(batch[6])
            targets_all.extend(out_label_ids.tolist())
            total_pred_list.extend(preds.tolist())
            idx += batch_size

    accuracy = 0
    for i in range(len(targets_all)):
        if total_pred_list[i] == targets_all[i]:
            accuracy += 1
    accuracy = accuracy / len(targets_all)

    loss_all = loss_all.cpu()
    loss_all = (
        (loss_all - loss_all.min()) / (loss_all.max() - loss_all.min())
    )
    loss_all = loss_all.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(loss_all)
    prob = gmm.predict_proba(loss_all)
    prob = prob[:, gmm.means_.argmin()]
    chosen_idx_all_gmm = np.where(prob > args["TAU"])[0]
    for idx in chosen_idx_all_gmm:
        chosen_list.append(raw_index_list[idx])

    
    logger.info(f"Validation Accuracy with all data (noisy data) {batch_idx}/{epoch}: {accuracy:.4f}")
    logger.info(f"gmm choose {len(chosen_list)} samples")
    return accuracy, raw_index_list, total_pred_list, chosen_list, targets_all  


def validate_on_labeled_gt(raw_index_list, total_pred_list, labeled_gt_path, logger):
    labels_dict = {}
    for i in range(len(raw_index_list)):
        labels_dict[raw_index_list[i]] = total_pred_list[i]

    labeled_gt_dict = {}
    with open(labeled_gt_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            labeled_gt_dict[row[0]] = int(row[1])

    correct_cnt = 0
    total_cnt = 0
    for idx in labeled_gt_dict.keys():
        if idx not in labels_dict.keys():
            continue
        now_label = labels_dict[idx]
        train_label = labeled_gt_dict[idx]
        if now_label == train_label:
            correct_cnt += 1
        total_cnt += 1

    logger.info(f"validate_on_labeled_gt: {correct_cnt/total_cnt:.4f}, correct_cnt:{correct_cnt}, total_cnt:{total_cnt}")
    return correct_cnt/total_cnt, correct_cnt, total_cnt


def validate_chosen_list_on_labeled_gt(chosen_index_list, raw_index_list, targets_all, labeled_gt_path, logger):
    labels_dict = {}
    for i in range(len(raw_index_list)):
        labels_dict[raw_index_list[i]] = targets_all[i]

    labeled_gt_dict = {}
    with open(labeled_gt_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            labeled_gt_dict[row[0]] = int(row[1])

    chosen_list_dict = {}
    for i in range(len(chosen_index_list)):
        chosen_list_dict[chosen_index_list[i]] = labels_dict[chosen_index_list[i]]

    correct_cnt = 0
    total_cnt = 0
    for idx in labeled_gt_dict.keys():
        if idx not in chosen_list_dict.keys():
            continue
        now_label = chosen_list_dict[idx]
        train_label = labeled_gt_dict[idx]
        if now_label == train_label:
            correct_cnt += 1
        total_cnt += 1

    logger.info(f"gmm chosen list: {correct_cnt/total_cnt:.4f}, correct_cnt:{correct_cnt}, total_cnt:{total_cnt}")

    return correct_cnt/total_cnt, correct_cnt, total_cnt


def  train(
    task="mmimdb",
    train_file_path="",
    dev_file_path="",
    model_save_path="",
    log_dir="",
    confidence_threshold=0.9,
    num_labels=None
):
    args = get_args(task)
    logger = set_logger(args, log_dir)
    set_seed(args["SEED"])

    labels = get_labels_from_label_num(args["NUM_LABELS"])
    num_labels = len(labels)
    transformer_config = AutoConfig.from_pretrained(
        "google-bert/bert-base-uncased",
        cache_dir=args["CACHE_DIR"],
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-uncased",
        do_lower_case=True,
        cache_dir=args["CACHE_DIR"],
        local_files_only=True,
    )
    transformer = AutoModel.from_pretrained(
        "google-bert/bert-base-uncased", config=transformer_config, cache_dir=args["CACHE_DIR"],
        local_files_only=True,
    )
    img_encoder = ImageEncoder(args)
    config = MMBTConfig(transformer_config, num_labels=num_labels)

    try:
        model = MMBTForClassification(config, transformer, img_encoder)
    except Exception as e:
        raise Exception(f"Error loading MMBTForClassification: {e} MMBT will be deprecated in the future, still available for transformer 4.50.3")

    model.to(args["DEVICE"])
    all_dataset = CsvDataset(
        csv_data_path=train_file_path,
        tokenizer=tokenizer,
        transforms=get_image_transforms(),
        image_base_dir=args["IMAGE_DIR"],
        max_seq_length=args["max_seq_length"] - args["num_image_embeds"] - 2,
        num_labels=args["NUM_LABELS"],
        mode = task,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    train_batch_size = 8
    train_sampler = RandomSampler(all_dataset)
    all_dataloader = DataLoader(
        all_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
        collate_fn=collate_fn,
    )
    t_total = len(all_dataloader) * args["mmbt_epoches"]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=0.00002, eps=0.00000001)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args["warmup_port"] * len(all_dataloader), 
        num_training_steps=t_total
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(all_dataset))
    logger.info("  Num Epochs = %d", args["mmbt_epoches"])
    model.zero_grad()
    set_seed(args["SEED"])

    if not os.path.exists(os.path.join(model_save_path, "checkpoint-best-train")):
        os.makedirs(os.path.join(model_save_path, "checkpoint-best-train"))
    if not os.path.exists(os.path.join(model_save_path, "checkpoint-best-val")):
        os.makedirs(os.path.join(model_save_path, "checkpoint-best-val"))


    best_acc_train = 0
    best_acc_val = 0
    chosen_index_list = []

    for epoch in range(args["mmbt_epoches"]):
        model.train()
        if epoch >= args["warmup_port"]:
            logger.info("warmup ends, selection begin")
            logger.info("length of chosen_index_list: %d", len(chosen_index_list))
            
        total_loss = 0
        trained_count = 0
        for step, batch in tqdm(enumerate(all_dataloader), total=len(all_dataloader)):
            raw_index = batch[6]
            util.append_dict_to_yml(os.path.join(model_save_path, "extra_info.yml"), {"progress": (epoch*len(all_dataloader)+step)/(len(all_dataloader)*args["mmbt_epoches"])})
            if epoch >= args["warmup_port"]:
                skip_flag = True
                for i in range(len(raw_index)):
                    if raw_index[i] in chosen_index_list:
                        skip_flag = False
                        break
                if skip_flag:
                    continue

            labels = batch[5].to(args["DEVICE"])
            inputs = {
                "input_ids": batch[0].to(args["DEVICE"]),
                "input_modal": batch[2].to(args["DEVICE"]),
                "attention_mask": batch[1].to(args["DEVICE"]),
                "modal_start_tokens": batch[3].to(args["DEVICE"]),
                "modal_end_tokens": batch[4].to(args["DEVICE"]),
                "return_dict": False
            }
            outputs = model(**inputs)
            logits = outputs[0]

            if epoch >= args["warmup_port"]:
                chosen_mask = torch.zeros(len(raw_index)).to(args["DEVICE"])
                for i in range(len(raw_index)):
                    if raw_index[i] in chosen_index_list:
                        chosen_mask[i] = 1
                    else:
                        chosen_mask[i] = 0
            else:
                chosen_mask = torch.ones(len(raw_index)).to(args["DEVICE"])
            trained_count += chosen_mask.sum().item()
            
            loss = criterion(logits[chosen_mask == 1], labels[chosen_mask == 1])
            if torch.isnan(loss):
                logger.info(chosen_mask)
                logger.info(raw_index)
                logger.info(labels)
                logger.info(labels[chosen_mask == 1])
                logger.info(logits[chosen_mask == 1])
                raise ValueError("loss is nan")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            logger.info(f"Epoch {epoch}/{args['mmbt_epoches']} Batch {step}/{len(all_dataloader)} Loss: {loss.item():.4f}")

        logger.info("trained_count: %d", trained_count)
        if epoch >= args["warmup_port"]:
            if trained_count != len(chosen_index_list):
                raise ValueError("trained_count != len(chosen_index_list)")
            
        logger.info("validating...")
        model.eval()
        acc_on_train, raw_index_list, total_pred_list, chosen_index_list, targets_all = evaluate_model(args, model, all_dataloader, 99, epoch, logger)
        acc_on_labeled_gt, right_cnt, total_cnt = validate_on_labeled_gt(raw_index_list, total_pred_list, dev_file_path, logger)
        chosen_acc_on_labeled_gt, chosen_right_cnt, chosen_total_cnt = validate_chosen_list_on_labeled_gt(chosen_index_list, raw_index_list, targets_all, dev_file_path, logger)

        
        if acc_on_train > best_acc_train:
            best_acc_train = acc_on_train
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )
            torch.save(model_to_save.state_dict(), os.path.join(model_save_path, "checkpoint-best-train", WEIGHTS_NAME))
            logger.info("Saving model checkpoint to %s", os.path.join(model_save_path, "checkpoint-best-train", WEIGHTS_NAME))
            with open(os.path.join(model_save_path, "checkpoint-best-train", "acc.txt"), "w") as f:
                f.write(f"best_acc_train:{best_acc_train:.4f}, acc_on_labeled_gt:{acc_on_labeled_gt:.4f}, chosen_acc_on_labeled_gt:{chosen_acc_on_labeled_gt:.4f}, right_cnt:{right_cnt}, total_cnt:{total_cnt}, chosen_right_cnt:{chosen_right_cnt}, chosen_total_cnt:{chosen_total_cnt}\n")

        if acc_on_labeled_gt > best_acc_val:
            best_acc_val = acc_on_labeled_gt
            model_to_save = (
                model.module if hasattr(model, "module") else model
            ) 
            torch.save(model_to_save.state_dict(), os.path.join(model_save_path, "checkpoint-best-val", WEIGHTS_NAME))
            logger.info("Saving model checkpoint to %s", os.path.join(model_save_path, "checkpoint-best-val", WEIGHTS_NAME))
            with open(os.path.join(model_save_path, "checkpoint-best-val", "acc.txt"), "w") as f:
                f.write(f"best_acc_val:{best_acc_val:.4f}, acc_on_train:{acc_on_train:.4f}, chosen_acc_on_labeled_gt:{chosen_acc_on_labeled_gt:.4f}, right_cnt:{right_cnt}, total_cnt:{total_cnt}, chosen_right_cnt:{chosen_right_cnt}, chosen_total_cnt:{chosen_total_cnt}\n")

    util.append_dict_to_yml(os.path.join(model_save_path, "extra_info.yml"), {"progress": 1})
