import csv
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import copy
import yaml
import os
import logging
from datetime import datetime
import random
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import numpy as np
from transformers import get_scheduler
from src.utils import util


class RobertaDataset(Dataset):
    def __init__(self, csv_file_path, tokenizer, max_length=512, skip_header=True, confidence_threshold=0, mode=None):
        self.samples = []
        self.labels = []
        self.raw_index = []
        self.inputs = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            if skip_header:
                next(reader)
            for row in reader:
                if confidence_threshold > 0 and float(row[4]) < confidence_threshold:
                    continue
                self.samples.append(row[2])
                self.labels.append(int(row[1]))
                self.raw_index.append(row[0])

                if self.mode == "vsnli":
                    premise, hypothesis = row[2].split("#####")
                    inputs = self.tokenizer(
                        premise,
                        hypothesis,
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                else:
                    inputs = self.tokenizer(
                        row[2],
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                self.inputs.append(inputs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        label = self.labels[idx]
        raw_index = self.raw_index[idx]
        inputs = self.inputs[idx]

        return {
            "inputs": {key: val.squeeze(0) for key, val in inputs.items()},
            "labels": torch.tensor(label, dtype=torch.long),
            "text": text,
            "raw_index": raw_index
        }

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
    args["DATA_ROOT"] = os.path.join(args["PROJECT_ROOT"], "data")
    args["OUTPUT_DIR"] = os.path.join(args["PROJECT_ROOT"], "output", task)
    return args


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_logger(args, log_dir, name="roberta"):
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


def evaluate_model(args, model, val_dataloader, device, batch_idx, epoch, logger):
    model.eval()
    total_correct = 0
    total_samples = 0
    targets_all = []
    raw_index_list = []
    chosen_list = []
    total_pred_list = []
    loss_all = torch.zeros((len(val_dataloader.dataset))).to(args["DEVICE"])

    idx = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = logits.argmax(-1)
            loss = F.cross_entropy(logits, labels)
            batch_size = labels.size(0)
            loss_all[idx:idx+batch_size] = loss.detach()

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            raw_index_list.extend(batch['raw_index'])
            total_pred_list.extend(predictions.cpu().tolist())
            targets_all.extend(labels.cpu().tolist())
            idx += batch_size
    
    accuracy = total_correct / total_samples

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




def train(
    task="mmimdb",
    train_file_path="",
    dev_file_path="",
    model_save_path="",
    log_dir="",
    confidence_threshold=0,
    num_labels=2
):
    args = get_args(task)
    logger = set_logger(args, log_dir)
    set_seed(args["SEED"])
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", cache_dir=args["CACHE_DIR"], local_files_only=True)
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels, cache_dir=args["CACHE_DIR"], local_files_only=True)

    all_dataset = RobertaDataset(
        csv_file_path=train_file_path,
        tokenizer=tokenizer,
        max_length=512,
        skip_header=True,
        confidence_threshold=0,
        mode=task,
    )
    
    logger.info(f"Train dataset size: {len(all_dataset)}")


    all_dataloader = DataLoader(
        all_dataset,
        batch_size=args["roberta_batch"],
        shuffle=False
    )

    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    num_training_steps = len(all_dataloader) * args["roberta_epoches"]
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args["warmup_port"] * len(all_dataloader),
        num_training_steps=num_training_steps
    )

    device = args["DEVICE"]
    model.to(device)
    
    epochs = args["roberta_epoches"]
    model.train()
    best_acc_train = 0
    best_acc_val = 0

    chosen_index_list = []
    
    for epoch in range(epochs):
        model.train()
        if epoch >= args["warmup_port"]:
            logger.info("warmup ends, selection begin")
            logger.info("length of chosen_index_list: %d", len(chosen_index_list))
        total_loss = 0
        trained_count = 0
        for batch_idx, batch in enumerate(tqdm(all_dataloader)):
            util.append_dict_to_yml(os.path.join(model_save_path, "extra_info.yml"), {"progress": (epoch*len(all_dataloader)+batch_idx)/(len(all_dataloader)*epochs)})
            raw_index = batch["raw_index"]

            if epoch >= args["warmup_port"]:
                skip_flag = True
                for i in range(len(raw_index)):
                    if raw_index[i] in chosen_index_list:
                        skip_flag = False
                        break
                if skip_flag:
                    continue

            inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
            labels = batch["labels"].to(device)

            outputs = model(**inputs)
            logits = outputs.logits

            if epoch >= args["warmup_port"]:
                chosen_mask = torch.zeros(len(raw_index)).to(device)
                for i in range(len(raw_index)):
                    if raw_index[i] in chosen_index_list:
                        chosen_mask[i] = 1
                    else:
                        chosen_mask[i] = 0
            else:
                chosen_mask = torch.ones(len(raw_index)).to(device)

            trained_count += chosen_mask.sum().item()

            loss = F.cross_entropy(logits[chosen_mask == 1], labels[chosen_mask == 1])
            if torch.isnan(loss):
                logger.info(chosen_mask)
                logger.info(raw_index)
                logger.info(labels)
                logger.info(labels[chosen_mask == 1])
                logger.info(logits[chosen_mask == 1])
                raise ValueError("loss is nan")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(all_dataloader)}, Loss: {loss.item():.4f}")

        logger.info("trained_count: %d", trained_count)
        if epoch >= args["warmup_port"]:
            if trained_count != len(chosen_index_list):
                raise ValueError("trained_count != len(chosen_index_list)")
        
        avg_loss = total_loss / len(all_dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        logger.info("validating...")
        model.eval()

        acc_on_train, raw_index_list, total_pred_list, chosen_index_list, targets_all = evaluate_model(args, model, all_dataloader, args["DEVICE"], 99, epoch, logger)
        acc_on_labeled_gt, right_cnt, total_cnt = validate_on_labeled_gt(raw_index_list, total_pred_list, dev_file_path, logger)
        chosen_acc_on_labeled_gt, chosen_right_cnt, chosen_total_cnt = validate_chosen_list_on_labeled_gt(chosen_index_list, raw_index_list, targets_all, dev_file_path, logger)
        if acc_on_train > best_acc_train:
            best_acc_train = acc_on_train
            model.save_pretrained(os.path.join(model_save_path, "checkpoint-best-train"))
            with open(os.path.join(model_save_path, "checkpoint-best-train", "acc.txt"), "w") as f:
                f.write(f"best_acc_train:{best_acc_train:.4f}, acc_on_labeled_gt:{acc_on_labeled_gt:.4f}, chosen_acc_on_labeled_gt:{chosen_acc_on_labeled_gt:.4f}, right_cnt:{right_cnt}, total_cnt:{total_cnt}, chosen_right_cnt:{chosen_right_cnt}, chosen_total_cnt:{chosen_total_cnt}\n")

        if acc_on_labeled_gt > best_acc_val:
            best_acc_val = acc_on_labeled_gt
            model.save_pretrained(os.path.join(model_save_path, "checkpoint-best-val"))
            with open(os.path.join(model_save_path, "checkpoint-best-val", "acc.txt"), "w") as f:
                f.write(f"best_acc_val:{best_acc_val:.4f}, acc_on_train:{acc_on_train:.4f}, chosen_acc_on_labeled_gt:{chosen_acc_on_labeled_gt:.4f}, right_cnt:{right_cnt}, total_cnt:{total_cnt}, chosen_right_cnt:{chosen_right_cnt}, chosen_total_cnt:{chosen_total_cnt}\n")

    util.append_dict_to_yml(os.path.join(model_save_path, "extra_info.yml"), {"progress": 1})