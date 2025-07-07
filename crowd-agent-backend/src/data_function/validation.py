import src.data_processor.image_text_classification_processor_str_index as image_text_classification_processor
import src.utils.util as util
import os
import csv
from crowdlib.quality.confidence.classification.bayes import SingleChoiceBayesInference
import yaml
from pprint import pprint
import logging
import evaluate


def get_error_matrix_dict(target_set, ground_truth_set, chosen_idx_list, num_labels=2):
    chosen_target_dict = {}
    chosen_ground_truth_dict = {}
    for sample in target_set:
        if sample.index in chosen_idx_list:
            chosen_target_dict[sample.index] = sample
    for sample in ground_truth_set:
        if sample.index in chosen_idx_list:
            chosen_ground_truth_dict[sample.index] = sample

    error_matrix = {}
    for i in range(num_labels):
        error_matrix[i] = {}
        for j in range(num_labels):
            error_matrix[i][j] = 0

    for idx in chosen_idx_list:
        target_sample = chosen_target_dict[idx]
        ground_truth_sample = chosen_ground_truth_dict[idx]
        error_matrix[target_sample.label][ground_truth_sample.label] += 1

    total_count = 0
    for i in range(num_labels):
        for j in range(num_labels):
            total_count += error_matrix[i][j]
    
    if total_count != len(chosen_idx_list):
        raise ValueError("The number of lines in the labeled file and the ground truth file are not the same")
    
    error_matrix_dict = {}
    for i in range(num_labels):
        for j in range(num_labels):
            error_matrix_dict[f"w_{i}_t_{j}"] = error_matrix[i][j]
    
    return error_matrix_dict



def val_output_confusion_matrix(labeled_file, ground_truth_file, worker_name, confusion_matrix_file, num_labels=2):
    processor = (
        image_text_classification_processor.ImageTextClassificationProcessor()
    )
    labeled_set = processor.get_train_examples(labeled_file, skip_head=True)
    if len(labeled_set) == 0:
        logging.warning(f"labeled_file {labeled_file} is empty")
        return
    ground_truth_set = processor.get_train_examples(ground_truth_file, skip_head=True)

    labeled_sample_idx_list = [sample.index for sample in labeled_set]
    ground_truth_sample_idx_list = [sample.index for sample in ground_truth_set]
    chosen_idx_list = set(labeled_sample_idx_list) & set(ground_truth_sample_idx_list)

    error_matrix_dict = get_error_matrix_dict(labeled_set, ground_truth_set, chosen_idx_list, num_labels)

    worker_data = {"total_cnt": len(chosen_idx_list)}
    
    for i in range(num_labels):
        for j in range(num_labels):
            key = f"w_{i}_t_{j}"
            worker_data[key] = error_matrix_dict[key]
    
    dict_data = {worker_name: worker_data}
    util.append_dict_to_yml(confusion_matrix_file, dict_data)


def val_previous_output_confusion_matrix(labeled_file, last_round_aggregated_file, ground_truth_file, worker_name, confusion_matrix_file):
    processor = (
        image_text_classification_processor.ImageTextClassificationProcessor()
    )
    labeled_set = processor.get_train_examples(labeled_file, skip_head=True)
    last_round_aggregated_set = processor.get_train_examples(last_round_aggregated_file, skip_head=True)
    if len(labeled_set) == 0:
        logging.warning(f"labeled_file {labeled_file} is empty")
        return
    ground_truth_set = processor.get_train_examples(ground_truth_file, skip_head=True)

    labeled_sample_idx_list = [sample.index for sample in labeled_set]
    ground_truth_sample_idx_list = [sample.index for sample in ground_truth_set]
    chosen_idx_list = set(labeled_sample_idx_list) & set(ground_truth_sample_idx_list)
    
    error_matrix_dict = get_error_matrix_dict(last_round_aggregated_set, ground_truth_set, chosen_idx_list)

    dict_data = {
        worker_name: {
            "w_0_t_0": error_matrix_dict["w_0_t_0"],
            "w_0_t_1": error_matrix_dict["w_0_t_1"],
            "w_1_t_0": error_matrix_dict["w_1_t_0"],
            "w_1_t_1": error_matrix_dict["w_1_t_1"],
            "total_cnt": len(chosen_idx_list)
        }
    }
    util.append_dict_to_yml(confusion_matrix_file, dict_data)


def read_confusion_matrix(confusion_matrix, worker_name, num_labels=2):
    matrix_counts = {}
    for i in range(num_labels):
        for j in range(num_labels):
            key = f"w_{i}_t_{j}"
            matrix_counts[(i, j)] = confusion_matrix[worker_name][key]
    
    total_cnt = confusion_matrix[worker_name]["total_cnt"]
    
    correct_counts = 0
    for i in range(num_labels):
        correct_counts += matrix_counts[(i, i)]
    
    avg_acc = correct_counts / total_cnt if total_cnt != 0 else 1/num_labels
    
    error_matrix = {}
    for i in range(num_labels):
        error_matrix[i] = {}
        
        row_sum = 0
        for j in range(num_labels):
            row_sum += matrix_counts[(j, i)]
        
        for j in range(num_labels):
            if row_sum > 6 and total_cnt > 18:
                if i == j:
                    error_matrix[i][j] = matrix_counts[(j, i)] / row_sum
                else:
                    error_matrix[i][j] = matrix_counts[(j, i)] / row_sum
            else:
                if i == j:
                    error_matrix[i][j] = avg_acc
                else:
                    error_matrix[i][j] = (1 - avg_acc) / (num_labels - 1)
    
    if "human" not in worker_name:
        for i in range(num_labels):
            for j in range(num_labels):
                if error_matrix[i][j] == 0:
                    error_matrix[i][j] = 1e-10
                elif error_matrix[i][j] == 1:
                    error_matrix[i][j] = 1-1e-10

    return error_matrix


def tag_aggregation_bayes(output_task_dir, tag_aggregation_file, train_path, round_limit=99, num_labels=2):
    confusion_matrix_yml_path = os.path.join(output_task_dir, "confusion_matrix.yml")
    with open(confusion_matrix_yml_path, "r") as f:
        confusion_matrix = yaml.load(f, Loader=yaml.FullLoader)

    processor = (
        image_text_classification_processor.ImageTextClassificationProcessor()
    )

    labeled_samples = {}
    worker_name_list = []

    for worker_name in confusion_matrix.keys():
        loop_cnt = worker_name.split("_")[0]
        if int(loop_cnt) > round_limit:
            continue
        worker_name_list.append(worker_name)
        error_matrix = read_confusion_matrix(confusion_matrix, worker_name, num_labels)
        pprint(worker_name)
        pprint(error_matrix)
    
        agent = "_".join(worker_name.split("_")[1:])
        labeled_file = os.path.join(output_task_dir, f"round{loop_cnt}", f"{agent}.csv")
        labeled_set = processor.get_train_examples(labeled_file, skip_head=True)
        
        for sample in labeled_set:
            if sample.index not in labeled_samples:
                labeled_samples[sample.index] = [(sample.label, error_matrix, worker_name)]
            else:
                labeled_samples[sample.index].append((sample.label, error_matrix, worker_name))

    aggregated_samples = {}
    for sample_idx, label_info_list in labeled_samples.items():
        label, prob = tag_aggregation_one_sample_error_matrix(label_info_list, num_labels)
        aggregated_samples[sample_idx] = (label, prob)

    worker_name_list = sorted(worker_name_list)
    original_samples = processor.get_train_examples(train_path, skip_head=True)

    with open(tag_aggregation_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "text", "image_path", "confidence"] + worker_name_list)
        for sample in original_samples:
            idx = sample.index
            label, prob = aggregated_samples[idx]
            worker_labels = []
            for worker_name in worker_name_list:
                is_find = False
                for label_info in labeled_samples[idx]:
                    if label_info[2] == worker_name:
                        worker_labels.append(label_info[0])
                        is_find = True
                        break
                if not is_find:
                    worker_labels.append("#")
            writer.writerow([idx, label, sample.contents, sample.image_path, prob] + worker_labels)


def tag_aggregation_one_sample(label_info_list):
    bayes = SingleChoiceBayesInference(num_choices=2)
    for label, acc, worker_name in label_info_list:
        choice_probas = bayes.fit_predict_prob(worker_choice=label, worker_accuracy=acc)
    max_label = max(choice_probas, key=choice_probas.get)
    max_prob = choice_probas[max_label]
    return max_label, max_prob


def tag_aggregation_one_sample_error_matrix(label_info_list, num_labels=2):
    bayes = SingleChoiceBayesInference(num_choices=num_labels)
    for label, custom_error_matrix, worker_name in label_info_list:
        if "human" in worker_name:
            return label, 1
        choice_probas = bayes.fit_predict_prob(worker_choice=label, worker_error_matrix=custom_error_matrix)
    max_label = max(choice_probas, key=choice_probas.get)
    max_prob = choice_probas[max_label]
    return max_label, max_prob


def get_val_acc(output_path, ground_truth_path):
    processor = (
        image_text_classification_processor.ImageTextClassificationProcessor()
    )
    output_set = processor.get_train_examples(output_path, skip_head=True)
    ground_truth_set = processor.get_train_examples(ground_truth_path, skip_head=True)
    output_sample_idx_list = [sample.index for sample in output_set]
    ground_truth_sample_idx_list = [sample.index for sample in ground_truth_set]

    chosen_idx_list = set(output_sample_idx_list) & set(ground_truth_sample_idx_list)
    chosen_output_set = {sample.index: sample for sample in output_set if sample.index in chosen_idx_list}
    chosen_ground_truth_set = {sample.index: sample for sample in ground_truth_set if sample.index in chosen_idx_list}

    acc = 0
    for output_sample in chosen_output_set.values():
        if output_sample.label == chosen_ground_truth_set[output_sample.index].label:
            acc += 1
    acc /= len(chosen_idx_list)
    return acc, len(chosen_idx_list)


def eval_results(all_str_labels, all_str_preds):
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=all_str_preds, references=all_str_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"], 
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]  
    }


def get_val_acc_list(sample_list, ground_truth_list):
    output_sample_idx_list = [sample[0] for sample in sample_list]
    ground_truth_sample_idx_list = [sample[0] for sample in ground_truth_list]

    chosen_idx_list = set(output_sample_idx_list) & set(ground_truth_sample_idx_list)
    chosen_output_set = {sample[0]: sample for sample in sample_list if sample[0] in chosen_idx_list}
    chosen_ground_truth_set = {sample[0]: sample for sample in ground_truth_list if sample[0] in chosen_idx_list}

    acc = 0
    for output_sample in chosen_output_set.values():
        if output_sample[1] == chosen_ground_truth_set[output_sample[0]][1]:
            acc += 1
    if len(chosen_idx_list) == 0:
        return 0, 0
    acc /= len(chosen_idx_list)
    return acc, len(chosen_idx_list)


def get_val_acc_with_confidence_threshold(output_path, ground_truth_path, confidence_threshold=0.9):
    processor = (
        image_text_classification_processor.ImageTextClassificationProcessor()
    )
    output_set = processor.get_train_examples(output_path, skip_head=True)
    ground_truth_set = processor.get_train_examples(ground_truth_path, skip_head=True)
    output_sample_idx_list = [sample.index for sample in output_set]
    ground_truth_sample_idx_list = [sample.index for sample in ground_truth_set]

    satisfied_idx_list = []
    with open(output_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if float(row[4]) >= confidence_threshold:
                satisfied_idx_list.append(row[0])
                
    chosen_idx_list = set(output_sample_idx_list) & set(ground_truth_sample_idx_list)
    chosen_idx_list = set(satisfied_idx_list) & set(chosen_idx_list)
    chosen_output_set = {sample.index: sample for sample in output_set if sample.index in chosen_idx_list}
    chosen_ground_truth_set = {sample.index: sample for sample in ground_truth_set if sample.index in chosen_idx_list}

    acc = 0
    for output_sample in chosen_output_set.values():
        if output_sample.label == chosen_ground_truth_set[output_sample.index].label:
            acc += 1
    acc /= len(chosen_idx_list)
    return acc, len(chosen_idx_list)