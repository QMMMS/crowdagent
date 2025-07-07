import os
import csv
import src.data_processor.classification_processor as classification_processor
import src.data_function.validation as validation
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime, timedelta
import numpy as np
import yaml

def check_dir(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def merge_back_translate_data(back_translate_file_path, labeled_file_path, output_file_path):
    processor = classification_processor.ClassificationProcessor()
    labeled_data = processor.get_train_examples(labeled_file_path)
    back_translate_data = processor.get_train_bt_examples(back_translate_file_path)
    if len(labeled_data) != len(back_translate_data):
        raise ValueError("labeled_data and back_translate_data have different length")
    
    with open(output_file_path, "w", encoding="utf-8", newline="") as f:
        for i in range(len(labeled_data)):
            if labeled_data[i].contents != back_translate_data[i].contents:
                raise ValueError("labeled_data and back_translate_data have different contents")
            writer = csv.writer(f)
            writer.writerow([labeled_data[i].label, back_translate_data[i].contents, back_translate_data[i].contents_bt])


def display_confidence_distribution(confidence_csv_path, confidence_threshold):
    with open(confidence_csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        data = [row for row in reader]

    count_confidence_under_threshold = 0
    count_dict = {
        "0~0.3": 0,
        "0.3~0.6": 0,
        "0.6~0.9": 0,
        "0.9~1": 0,
    }
    total_confidence = 0
    for row in data:
        confidence = float(row[4])
        total_confidence += confidence
        if 0 <= confidence < 0.3:
            count_dict["0~0.3"] += 1
        elif 0.3 <= confidence < 0.6:
            count_dict["0.3~0.6"] += 1
        elif 0.6 <= confidence < 0.9:
            count_dict["0.6~0.9"] += 1
        elif 0.9 <= confidence <= 1:
            count_dict["0.9~1"] += 1
        else:
            raise ValueError("confidence out of range")
        
    for row in data:
        confidence = float(row[4])
        if confidence < confidence_threshold:
            count_confidence_under_threshold += 1
        
    average_confidence = total_confidence / len(data)
    return count_dict, average_confidence, count_confidence_under_threshold

def display_confidence_distribution_ner(confidence_csv_path, confidence_threshold):
    with open(confidence_csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        data = [row for row in reader]

    count_confidence_under_threshold = 0
    count_dict = {
        "0~0.3": 0,
        "0.3~0.6": 0,
        "0.6~0.9": 0,
        "0.9~1": 0,
    }
    total_confidence = 0
    for row in data:
        confidence_list = eval(row[4])
        confidence = min(confidence_list)
        total_confidence += confidence
        if 0 <= confidence < 0.3:
            count_dict["0~0.3"] += 1
        elif 0.3 <= confidence < 0.6:
            count_dict["0.3~0.6"] += 1
        elif 0.6 <= confidence < 0.9:
            count_dict["0.6~0.9"] += 1
        elif 0.9 <= confidence <= 1:
            count_dict["0.9~1"] += 1
        else:
            raise ValueError("confidence out of range")
        
    for row in data:
        confidence_list = eval(row[4])
        confidence = min(confidence_list)
        if confidence < confidence_threshold:
            count_confidence_under_threshold += 1
        
    average_confidence = total_confidence / len(data)
    return count_dict, average_confidence, count_confidence_under_threshold


def get_high_confidence_indices(confidence_csv_path, confidence_threshold=0.9):
    with open(confidence_csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        data = [row for row in reader]

    high_confidence_indices = []
    for i in range(len(data)):
        raw_idx = data[i][0]
        confidence = float(data[i][4])
        if confidence > confidence_threshold:
            high_confidence_indices.append(raw_idx)

    return high_confidence_indices

def get_most_common_label(labels, weights):
    if len(labels) != len(weights):
        raise ValueError("labels and weights have different length")
    
    total_labels = []
    for i in range(len(labels)):
        if labels[i] == '#' or labels[i] == '-1':
            continue
        labels[i] = int(labels[i])
        total_labels.extend([labels[i]] * weights[i])

    return max(set(total_labels), key=total_labels.count)

def draw_tsne(embeddings, kmeans_labels, cluster_centers, num_clusters, img_save_path=None):
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    all_points = np.vstack([embeddings, cluster_centers])
    all_points_2d = tsne.fit_transform(all_points)

    embeddings_2d = all_points_2d[:len(embeddings)]
    centers_2d = all_points_2d[len(embeddings):]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=kmeans_labels, cmap='viridis', 
                         alpha=0.7, s=50, edgecolors='w', linewidths=0.5)
    
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
               c=range(num_clusters), cmap='viridis', 
               marker='*', s=200, edgecolors='k', linewidths=1.5)

    plt.colorbar(scatter, label='labels')
    plt.title('t-SNE visualization: Embeddings clustering results')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.tight_layout()
    plt.show()

    if img_save_path:
        plt.savefig(img_save_path, dpi=300)

def get_wrong_samples(part_annotataion_path, ground_truth_file_path):
    samples = []

    part_index_list = []
    with open(part_annotataion_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        part_annotataion_data = [row for row in reader]
        part_index_list = [row[0] for row in part_annotataion_data]

    if len(part_index_list) == 0:
        return samples
    ground_truth_data = []
    with open(ground_truth_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        ground_truth_data = [row for row in reader]
    ground_truth_dict = {row[0]: row[1] for row in ground_truth_data}
    for row in part_annotataion_data:
        if row[0] not in ground_truth_dict:
            continue
        if row[1] != ground_truth_dict[row[0]]:
            samples.append((row[0], row[2], row[1], ground_truth_dict[row[0]]))
    return samples

def evaluate_selected_samples(part_annotataion_path, last_aggregated_path, this_aggregated_path, labeled_gt_path, get_ave_conf=False):
    part_index_list = []
    with open(part_annotataion_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        part_annotataion_data = [row for row in reader]
        part_index_list = [row[0] for row in part_annotataion_data]

    if len(part_index_list) == 0:
        if get_ave_conf:
            return 0, 0, 0, 0, 0, 0
        else:
            return 0, 0, 0, 0

    last_aggregated_data = []
    last_average_conf = 0
    with open(last_aggregated_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        last_aggregated_data = [row for row in reader if row[0] in part_index_list]
        last_average_conf = sum([float(row[4]) for row in last_aggregated_data]) / len(last_aggregated_data)

    this_aggregated_data = []
    this_average_conf = 0
    with open(this_aggregated_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        this_aggregated_data = [row for row in reader if row[0] in part_index_list]
        this_average_conf = sum([float(row[4]) for row in this_aggregated_data]) / len(this_aggregated_data)

    ground_truth_data = []
    with open(labeled_gt_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        ground_truth_data = [row for row in reader]

    if len(last_aggregated_data) != len(this_aggregated_data):
        raise ValueError("last_aggregated_data and this_aggregated_data have different length")
    
    last_aggregated_acc, last_aggregated_cnt = validation.get_val_acc_list(last_aggregated_data, ground_truth_data)
    this_aggregated_acc, this_aggregated_cnt = validation.get_val_acc_list(this_aggregated_data, ground_truth_data)
    part_acc, part_cnt = validation.get_val_acc_list(part_annotataion_data, ground_truth_data)

    if last_aggregated_cnt != this_aggregated_cnt:
        raise ValueError("last_aggregated_cnt and this_aggregated_cnt have different value")
    
    if last_aggregated_cnt != part_cnt:
        raise ValueError("last_aggregated_cnt and part_cnt have different value")
    
    if get_ave_conf:
        return len(part_annotataion_data), last_aggregated_acc, part_acc, this_aggregated_acc, last_average_conf, this_average_conf
        
    return len(part_annotataion_data), last_aggregated_acc, part_acc, this_aggregated_acc

def count_clean_samples_and_acc(last_round_entropy_log_file_path, this_round_entropy_log_file_path, ground_truth_file_path):
    last_round_entropy_log_data = []
    with open(last_round_entropy_log_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        last_round_entropy_log_data = [row for row in reader]

    this_round_entropy_log_data = []
    with open(this_round_entropy_log_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        this_round_entropy_log_data = [row for row in reader]

    ground_truth_labels = []
    with open(ground_truth_file_path, 'r') as file:
        reader = csv.reader(file)
        ground_truth_labels = [row[0] for row in reader]

    if len(ground_truth_labels) != len(last_round_entropy_log_data):
        raise ValueError("ground_truth_labels and last_round_entropy_log_data have different length")
    if len(ground_truth_labels) != len(this_round_entropy_log_data):
        raise ValueError("ground_truth_labels and this_round_entropy_log_data have different length")
    
    clean_sample_idx = []

    for i in range(len(this_round_entropy_log_data)):
        if this_round_entropy_log_data[i][-2] != "#":
            clean_sample_idx.append(i)

    this_round_clean_samples_acc = 0
    for idx in clean_sample_idx:
        if this_round_entropy_log_data[idx][0] == ground_truth_labels[idx]:
            this_round_clean_samples_acc += 1

    this_round_clean_samples_acc = this_round_clean_samples_acc / len(clean_sample_idx)

    last_round_clean_samples_acc = 0
    for idx in clean_sample_idx:
        if last_round_entropy_log_data[idx][0] == ground_truth_labels[idx]:
            last_round_clean_samples_acc += 1

    last_round_clean_samples_acc = last_round_clean_samples_acc / len(clean_sample_idx)

    if last_round_clean_samples_acc != this_round_clean_samples_acc:
        raise ValueError("last_round_clean_samples_acc and this_round_clean_samples_acc have different value")
    
    this_round_clean_samples_entropy = 0
    for idx in clean_sample_idx:
        this_round_clean_samples_entropy += float(this_round_entropy_log_data[idx][3])

    this_round_clean_samples_entropy = this_round_clean_samples_entropy / len(clean_sample_idx)

    last_round_clean_samples_entropy = 0
    for idx in clean_sample_idx:
        last_round_clean_samples_entropy += float(last_round_entropy_log_data[idx][3])

    last_round_clean_samples_entropy = last_round_clean_samples_entropy / len(clean_sample_idx)

    return len(clean_sample_idx), this_round_clean_samples_acc, this_round_clean_samples_entropy, last_round_clean_samples_entropy

def write_info_to_yml(yml_file_path, key, value):
    if os.path.exists(yml_file_path):
        with open(yml_file_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
    else:
        data = {}

    data[key] = value

    with open(yml_file_path, 'w') as file:
        yaml.dump(data, file)

def read_info_from_yml(yml_file_path, key):
    try:
        with open(yml_file_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return data[key]
    except:
        return None

def append_dict_to_yml(yml_file_path, dict_data):
    if os.path.exists(yml_file_path):
        with open(yml_file_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
    else:
        data = {}

    data.update(dict_data)

    with open(yml_file_path, 'w') as file:
        yaml.dump(data, file)

def calculate_time_cost(yml_file_path):
    start_time = read_info_from_yml(yml_file_path, "start_time")
    end_time = read_info_from_yml(yml_file_path, "end_time")
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    time_cost = (end_time - start_time).total_seconds() / 3600
    return time_cost

def read_confusion_matrix(task_dir, round_cnt):
    with open(os.path.join(task_dir, "confusion_matrix.yml"), "r", encoding="utf8") as f:
        content = f.read()
        dict_content = yaml.load(content, Loader=yaml.FullLoader)
    ret = {}
    for key, value in dict_content.items():
        key_num = int(key.split("_")[0])
        if key_num == round_cnt:
            ret[key] = value
    return ret
    
