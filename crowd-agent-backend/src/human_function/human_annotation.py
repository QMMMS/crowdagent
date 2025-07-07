import pickle
import csv
import logging
import src.utils.util as util
import src.data_processor.image_text_classification_processor_str_index as image_text_classification_processor
import random
import torch
# from fuxi.aop.core import AOP
# from src.human_fuction import sdk
# import fuxi.aop.ddl.builtins as builtins
# from fuxi.aop.ddl.builtins import Image
# from fuxi.aop.edsl import State, Action, Observation, MemState
import os

def ground_truth_annotation(worst_idx_sel_path, train_file_path, ground_truth_file_path, output_file_path):
    with open(worst_idx_sel_path, "rb") as f:
        worst_idx_sel = pickle.load(f)

    with open(ground_truth_file_path, 'r') as f:
        ground_truth = list(csv.reader(f))

    with open(train_file_path, 'r') as f:
        train = list(csv.reader(f))

    copyed_train = train.copy()  
    for i in worst_idx_sel:
        copyed_train[i][0] = ground_truth[i][0]

    with open(output_file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(copyed_train)

    return copyed_train



def human_annotation_with_confidence(last_round_aggregated_file_path, output_file_path, unlabeled_gt_file_path, labeled_gt_file_path, confidence_threshold=0.9, annotation_rate=0.1):
    with open(last_round_aggregated_file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        last_round_aggregated = [row for row in reader]

    ground_truth_dict = {}
    processor = (
        image_text_classification_processor.ImageTextClassificationProcessor()
    )
    unlabeled_gt_set = processor.get_train_examples(unlabeled_gt_file_path, skip_head=True)
    labeled_gt_set = processor.get_train_examples(labeled_gt_file_path, skip_head=True)

    for sample in unlabeled_gt_set:
        ground_truth_dict[sample.index] = sample.label
    for sample in labeled_gt_set:
        ground_truth_dict[sample.index] = sample.label

    random.seed(42)
    random.shuffle(last_round_aggregated)
    last_round_aggregated.sort(key=lambda x: float(x[4]))

    less_than_confidence_threthold_count = 0
    for i in range(len(last_round_aggregated)):
        if float(last_round_aggregated[i][4]) < confidence_threshold:
            less_than_confidence_threthold_count += 1

    human_count = min(int(len(last_round_aggregated) * annotation_rate), less_than_confidence_threthold_count)
    human_annotation_list = []
    for i in range(human_count):
        idx = last_round_aggregated[i][0]
        true_label = ground_truth_dict[idx]
        human_annotation_list.append((idx, true_label, last_round_aggregated[i][2], last_round_aggregated[i][3]))

    with open(output_file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header[:4])
        writer.writerows(human_annotation_list)


def human_annotation_with_confidence_online(last_round_aggregated_file_path, to_be_annotated_file_path, confidence_threshold=0.9, annotation_rate=0.1):
    with open(last_round_aggregated_file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        last_round_aggregated = [row for row in reader]

    random.seed(42)
    random.shuffle(last_round_aggregated)
    last_round_aggregated.sort(key=lambda x: float(x[4]))

    less_than_confidence_threthold_count = 0
    for i in range(len(last_round_aggregated)):
        if float(last_round_aggregated[i][4]) < confidence_threshold:
            less_than_confidence_threthold_count += 1

    human_count = min(int(len(last_round_aggregated) * annotation_rate), less_than_confidence_threthold_count)

    util.append_dict_to_yml(os.path.join(os.path.dirname(to_be_annotated_file_path), "extra_info.yml"), {"label_count": human_count})

    human_annotation_list = []
    for i in range(human_count):
        idx = last_round_aggregated[i][0]
        line = [idx] + [" "] + last_round_aggregated[i][1:]
        human_annotation_list.append(line)

    with open(to_be_annotated_file_path  , 'w') as f:
        writer = csv.writer(f)
        human_header = [header[0]] + ["human_label"] + header[1:]
        writer.writerow(human_header)
        writer.writerows(human_annotation_list)


def human_annotation_with_coreset_online(last_round_aggregated_file_path, to_be_annotated_file_path, last_data_embeddings_path, confidence_threshold=0.9, annotation_rate=0.1):
    with open(last_round_aggregated_file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        last_round_aggregated = [row for row in reader]

    random.seed(42)
    random.shuffle(last_round_aggregated)
    last_round_aggregated.sort(key=lambda x: float(x[4]))

    less_than_confidence_threthold_count = 0
    for i in range(len(last_round_aggregated)):
        if float(last_round_aggregated[i][4]) < confidence_threshold:
            less_than_confidence_threthold_count += 1

    human_count = min(int(len(last_round_aggregated) * annotation_rate), less_than_confidence_threthold_count)
    
    util.append_dict_to_yml(os.path.join(os.path.dirname(to_be_annotated_file_path), "extra_info.yml"), {"label_count": human_count})


    if human_count * 2 <= less_than_confidence_threthold_count:
        existing_pool_nums = human_count * 2
    elif human_count * 2 > less_than_confidence_threthold_count:
        existing_pool_nums = less_than_confidence_threthold_count

    loaded_data = torch.load(last_data_embeddings_path)
    raw_idx_all = loaded_data['raw_idx_all']
    embeddings_all = loaded_data['embeddings_all']

    raw_idx_of_existing_pool = []
    for i in range(existing_pool_nums, len(last_round_aggregated)):
        raw_idx_of_existing_pool.append(last_round_aggregated[i][0])

    existing_pool = [raw_idx_all.index(i) for i in raw_idx_of_existing_pool]
    selected_idx = coreset_active_learning(embeddings_all, existing_pool, human_count)
    selected_raw_idx = [raw_idx_all[i] for i in selected_idx]


    human_annotation_list = []
    for sample in last_round_aggregated:
        if sample[0] in selected_raw_idx:
            idx = sample[0]
            line = [idx] + [" "] + sample[1:]
            human_annotation_list.append(line)

    with open(to_be_annotated_file_path, 'w') as f:
        writer = csv.writer(f)
        human_header = [header[0]] + ["human_label"] + header[1:]
        writer.writerow(human_header)
        writer.writerows(human_annotation_list)


def human_annotation_with_coreset(last_round_aggregated_file_path, output_file_path, unlabeled_gt_file_path, labeled_gt_file_path, last_data_embeddings_path, confidence_threshold=0.9, annotation_rate=0.1):
    with open(last_round_aggregated_file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        last_round_aggregated = [row for row in reader]

    ground_truth_dict = {}
    processor = image_text_classification_processor.ImageTextClassificationProcessor()
    unlabeled_gt_set = processor.get_train_examples(unlabeled_gt_file_path, skip_head=True)
    labeled_gt_set = processor.get_train_examples(labeled_gt_file_path, skip_head=True)

    for sample in unlabeled_gt_set:
        ground_truth_dict[sample.index] = sample.label
    for sample in labeled_gt_set:
        ground_truth_dict[sample.index] = sample.label

    random.seed(42)
    random.shuffle(last_round_aggregated)
    last_round_aggregated.sort(key=lambda x: float(x[4]))

    less_than_confidence_threthold_count = 0
    for i in range(len(last_round_aggregated)):
        if float(last_round_aggregated[i][4]) < confidence_threshold:
            less_than_confidence_threthold_count += 1

    human_count = min(int(len(last_round_aggregated) * annotation_rate), less_than_confidence_threthold_count)
    
    util.append_dict_to_yml(os.path.join(os.path.dirname(output_file_path), "extra_info.yml"), {"label_count": human_count})


    if human_count * 2 <= less_than_confidence_threthold_count:
        existing_pool_nums = human_count * 2
    elif human_count * 2 > less_than_confidence_threthold_count:
        existing_pool_nums = less_than_confidence_threthold_count

    loaded_data = torch.load(last_data_embeddings_path)
    raw_idx_all = loaded_data['raw_idx_all']
    embeddings_all = loaded_data['embeddings_all']

    raw_idx_of_existing_pool = []
    for i in range(existing_pool_nums, len(last_round_aggregated)):
        raw_idx_of_existing_pool.append(last_round_aggregated[i][0])

    existing_pool = [raw_idx_all.index(i) for i in raw_idx_of_existing_pool]
    selected_idx = coreset_active_learning(embeddings_all, existing_pool, human_count)
    selected_raw_idx = [raw_idx_all[i] for i in selected_idx]


    human_annotation_list = []
    for sample in last_round_aggregated:
        if sample[0] in selected_raw_idx:
            true_label = ground_truth_dict[sample[0]]
            human_annotation_list.append((sample[0], true_label, sample[2], sample[3]))

    with open(output_file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header[:4])
        writer.writerows(human_annotation_list)


# async def human_annotation_with_youling(last_round_aggregated_file_path, output_file_path, img_base_path, confidence_threshold=0.9, annotation_rate=0.1):
#     with open(last_round_aggregated_file_path, 'r') as f:
#         reader = csv.reader(f)
#         header = next(reader)
#         last_round_aggregated = [row for row in reader]

#     processor = (
#         image_text_classification_processor.ImageTextClassificationProcessor()
#     )

#     random.seed(42)
#     random.shuffle(last_round_aggregated)
#     last_round_aggregated.sort(key=lambda x: float(x[4]))

#     less_than_confidence_threthold_count = 0
#     for i in range(len(last_round_aggregated)):
#         if float(last_round_aggregated[i][4]) < confidence_threshold:
#             less_than_confidence_threthold_count += 1

#     human_count = min(min(int(len(last_round_aggregated) * annotation_rate), less_than_confidence_threthold_count),1)

#     aop = await AOP.init(task_type = sdk.Demo_Task, config = sdk.get_server_config("12602"))
#     agent = await aop.create_agent(sdk.Demo_Agent)

#     with open(output_file_path, 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(header[:4])

#     # for i in range(human_count):
#     i =1
#     raw_idx = last_round_aggregated[i][0]
#     text = last_round_aggregated[i][2]
#     img_path = os.path.join(img_base_path, last_round_aggregated[i][3])
#     image = await builtins.Image.from_path_async(img_path, 'jpg')
#     a = Observation[Image](image)
#     b = text
#     output = await agent.annotate(a, b)
#     label = output[0]['value']
#     with open(output_file_path, 'a') as f:
#         writer = csv.writer(f)
#         writer.writerow([raw_idx, label, text, img_path])


def coreset_active_learning(x: torch.Tensor, existing_pool: list, b: int) -> list:

    n = x.shape[0]
    selected = torch.zeros(n, dtype=bool)
    selected[existing_pool] = True
    
    if len(existing_pool) > 0:
        dists_to_pool = torch.cdist(x, x[existing_pool])
        min_dists = torch.min(dists_to_pool, dim=1)[0]
    else:
        min_dists = torch.full((n,), float('inf'))
    
    new_points = []
    
    for _ in range(b):
        candidates = (~selected).nonzero(as_tuple=False).view(-1)
        if candidates.numel() == 0:
            break
        min_dists_candidates = min_dists[candidates]
        u_idx = torch.argmax(min_dists_candidates)
        u = candidates[u_idx].item()
        new_points.append(u)
        selected[u] = True
        dists_to_u = torch.norm(x - x[u], dim=1)
        min_dists = torch.min(min_dists, dists_to_u)
    
    return new_points