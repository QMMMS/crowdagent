import csv
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import copy
import yaml
import os
import logging
from datetime import datetime
import random


class ChineseRobertaDataset(Dataset):
    def __init__(self, csv_file_path, tokenizer, max_length=512, skip_header=True, confidence_threshold=0):
        self.samples = []
        self.labels = []
        self.raw_index = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(csv_file_path, 'r') as f:
            reader = csv.reader(f)
            if skip_header:
                next(reader)
            for row in reader:
                if float(row[4]) < confidence_threshold:
                    continue
                self.samples.append(row[2])
                self.labels.append(int(row[1]))
                self.raw_index.append(int(row[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        label = self.labels[idx]
        raw_index = self.raw_index[idx]

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

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

def set_logger(args, log_dir, name="chinese-roberta"):
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


def validate_on_labeled_gt(raw_index_list, total_pred_list, labeled_gt_path, logger):
    labels_dict = {}
    for i in range(len(raw_index_list)):
        labels_dict[raw_index_list[i]] = total_pred_list[i]

    labeled_gt_dict = {}
    with open(labeled_gt_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            labeled_gt_dict[int(row[0])] = int(row[1])

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
            

def train(
    task="mr",
    train_file_path="",
    dev_file_path="",
    model_save_path="",
    log_dir="",
    confidence_threshold=0.9
):
    args = get_args(task)
    logger = set_logger(args, log_dir)
    set_seed(args["SEED"])
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", cache_dir=args["CACHE_DIR"], local_files_only=True)
    model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=2, cache_dir=args["CACHE_DIR"], local_files_only=True)

    train_dataset = ChineseRobertaDataset(
        csv_file_path=train_file_path,
        tokenizer=tokenizer,
        max_length=512,
        skip_header=True,
        confidence_threshold=confidence_threshold
    )
    logger.info(f"train_dataset length: {len(train_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args["roberta_batch"],
        shuffle=True
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    epochs = args["roberta_epoches"]
    model.train()
    model.to(args["DEVICE"])
    best_acc_train = 0
    best_acc_val = 0
    best_acc_val_train = 0
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            inputs = {k: v.to(args["DEVICE"]) for k, v in batch['inputs'].items()}
            labels = batch["labels"].to(args["DEVICE"])

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")


        logger.info("validating...")
        model.eval()
        acc_on_train, raw_index_list, total_pred_list = evaluate_model(model, copy.deepcopy(train_dataloader), args["DEVICE"], 99, epoch, logger)
        acc_on_labeled_gt, right_cnt, total_cnt = validate_on_labeled_gt(raw_index_list, total_pred_list, dev_file_path, logger)
        if acc_on_train > best_acc_train:
            best_acc_train = acc_on_train
            model.save_pretrained(os.path.join(model_save_path, "checkpoint-best-train"))
            with open(os.path.join(model_save_path, "checkpoint-best-train", "acc.txt"), "w") as f:
                f.write(f"best_acc_train:{best_acc_train:.4f}, acc_on_labeled_gt:{acc_on_labeled_gt:.4f}, right_cnt:{right_cnt}, total_cnt:{total_cnt}")

        if acc_on_labeled_gt > best_acc_val or (acc_on_labeled_gt == best_acc_val and acc_on_train > best_acc_val_train):
            best_acc_val = acc_on_labeled_gt
            best_acc_val_train = acc_on_train
            model.save_pretrained(os.path.join(model_save_path, "checkpoint-best-val"))
            with open(os.path.join(model_save_path, "checkpoint-best-val", "acc.txt"), "w") as f:
                f.write(f"acc_train:{acc_on_train:.4f}, best_acc_labeled_gt:{best_acc_val:.4f}, right_cnt:{right_cnt}, total_cnt:{total_cnt}")
        model.train()



def evaluate_model(model, val_dataloader, device, batch_idx, epoch, logger):
    model.eval()
    total_correct = 0
    total_samples = 0
    raw_index_list = []
    total_pred_list = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(-1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            raw_index_list.extend(batch['raw_index'].cpu().tolist())
            total_pred_list.extend(predictions.cpu().tolist())
    
    accuracy = total_correct / total_samples

    logger.info(f"Validation Accuracy {batch_idx}/{epoch}: {accuracy:.4f}")
    return accuracy, raw_index_list, total_pred_list
