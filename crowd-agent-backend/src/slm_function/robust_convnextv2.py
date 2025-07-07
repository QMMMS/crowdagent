import csv
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import copy
import os
import torch.nn.functional as F
import logging
import yaml
from datetime import datetime
import random
from PIL import Image
from sklearn.mixture import GaussianMixture
import numpy as np
from src.utils import util


class ConvNextV2Dataset(Dataset):
    def __init__(self, csv_file_path, processor, base_path, skip_header=True, confidence_threshold=0):
        self.image_paths = []
        self.labels = []
        self.raw_index = []
        self.processor = processor
        self.base_path = base_path

        with open(csv_file_path, 'r') as f:
            reader = csv.reader(f)
            if skip_header:
                next(reader)
            for row in reader:
                if confidence_threshold > 0 and float(row[4]) < confidence_threshold:
                    continue
                self.image_paths.append(row[3])
                self.labels.append(int(row[1]))
                self.raw_index.append(row[0])

        self.image_path_to_inputs = {}
        for image_path in tqdm(self.image_paths):
            full_path = os.path.join(self.base_path, image_path)
            image = Image.open(full_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            self.image_path_to_inputs[image_path] = inputs
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        raw_index = self.raw_index[idx]
        return {
            'inputs': self.image_path_to_inputs[image_path],
            'labels': torch.tensor(label, dtype=torch.long),
            'image_path': image_path,
            'raw_index': raw_index
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
    args["IMAGE_DIR"] = os.path.join(args["PROJECT_ROOT"], "data", task, "images")
    args["OUTPUT_DIR"] = os.path.join(args["PROJECT_ROOT"], "output", task)
    return args


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_logger(args, log_dir, name="convnextv2"):
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
        for batch in val_dataloader:
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
    if len(chosen_idx_all_gmm) < 500:
        logger.info(f"gmm choose {len(chosen_idx_all_gmm)} samples, less than 500, expand to 500")
        all_probs = prob
        all_indices = np.arange(len(all_probs))
        prob_index_pairs = list(zip(all_probs, all_indices))
        prob_index_pairs.sort(reverse=True)
        chosen_idx_all_gmm = np.array([idx for _, idx in prob_index_pairs[:500]])
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
    task="mr",
    train_file_path="",
    dev_file_path="",
    model_save_path="",
    log_dir="",
    confidence_threshold=0.9,
    id_to_label=None
):
    args = get_args(task)
    logger = set_logger(args, log_dir)
    set_seed(args["SEED"])

    processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-22k-384", cache_dir=args["CACHE_DIR"], local_files_only=True)
    id2label = id_to_label
    label2id = {v: k for k, v in id2label.items()}

    train_dataset = ConvNextV2Dataset(
        train_file_path, 
        processor, 
        args["IMAGE_DIR"], 
        confidence_threshold=0
    )
    logger.info(f"train_dataset length: {len(train_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,  
        shuffle=False,
    )

    model = AutoModelForImageClassification.from_pretrained(
        "facebook/convnextv2-tiny-22k-384", 
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes = True,
        cache_dir=args["CACHE_DIR"],
        local_files_only=True,
    )

    for param in model.convnextv2.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True
    optimizer = AdamW(model.classifier.parameters(), lr=1e-4)
    num_epochs = args["convnextv2_epoches"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args["warmup_port"] * len(train_dataloader), 
        num_training_steps=num_epochs
    )

    model.to(args["DEVICE"])
    model.train()

    best_acc_train = 0
    best_acc_val_train = 0
    best_acc_val = 0

    chosen_index_list = []

    for epoch in range(num_epochs):
        model.train()
        if epoch >= args["warmup_port"]:
            logger.info("warmup ends, selection begin")
            logger.info("length of chosen_index_list: %d", len(chosen_index_list))
        total_loss = 0
        trained_count = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            raw_index = batch["raw_index"]
            util.append_dict_to_yml(os.path.join(model_save_path, "extra_info.yml"), {"progress": (epoch*len(train_dataloader)+batch_idx)/(len(train_dataloader)*num_epochs)})

            if epoch >= args["warmup_port"]:
                skip_flag = True
                for i in range(len(raw_index)):
                    if raw_index[i] in chosen_index_list:
                        skip_flag = False
                        break
                if skip_flag:
                    continue

            inputs = {k: v.to(args["DEVICE"]) for k, v in batch['inputs'].items()}
            labels = batch['labels'].to(args["DEVICE"])

            outputs = model(**inputs)
            logits = outputs.logits

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
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        logger.info("trained_count: %d", trained_count)
        if epoch >= args["warmup_port"]:
            if trained_count != len(chosen_index_list):
                raise ValueError("trained_count != len(chosen_index_list)")
        
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        logger.info("validating...")
        model.eval()

        acc_on_train, raw_index_list, total_pred_list, chosen_index_list, targets_all = evaluate_model(args, model, train_dataloader, args["DEVICE"], 99, epoch, logger)
        acc_on_labeled_gt, right_cnt, total_cnt = validate_on_labeled_gt(raw_index_list, total_pred_list, dev_file_path, logger)
        chosen_acc_on_labeled_gt, chosen_right_cnt, chosen_total_cnt = validate_chosen_list_on_labeled_gt(chosen_index_list, raw_index_list, targets_all, dev_file_path, logger)
        if acc_on_train > best_acc_train:
            best_acc_train = acc_on_train
            model.save_pretrained(os.path.join(model_save_path, "checkpoint-best-train"))
            with open(os.path.join(model_save_path, "checkpoint-best-train", "acc.txt"), "w") as f:
                f.write(f"acc_on_train:{acc_on_train:.4f}, acc_on_labeled_gt:{acc_on_labeled_gt:.4f}, chosen_acc_on_labeled_gt:{chosen_acc_on_labeled_gt:.4f}, right_cnt:{right_cnt}, total_cnt:{total_cnt}, chosen_right_cnt:{chosen_right_cnt}, chosen_total_cnt:{chosen_total_cnt}")

        if acc_on_labeled_gt > best_acc_val or (acc_on_labeled_gt == best_acc_val and acc_on_train > best_acc_val_train):
            best_acc_val = acc_on_labeled_gt
            best_acc_val_train = acc_on_train
            model.save_pretrained(os.path.join(model_save_path, "checkpoint-best-val"))
            with open(os.path.join(model_save_path, "checkpoint-best-val", "acc.txt"), "w") as f:
                f.write(f"acc_train:{acc_on_train:.4f}, acc_labeled_gt:{acc_on_labeled_gt:.4f}, chosen_acc_labeled_gt:{chosen_acc_on_labeled_gt:.4f}, right_cnt:{right_cnt}, total_cnt:{total_cnt}, chosen_right_cnt:{chosen_right_cnt}, chosen_total_cnt:{chosen_total_cnt}")

    util.append_dict_to_yml(os.path.join(model_save_path, "extra_info.yml"), {"progress": 1})
