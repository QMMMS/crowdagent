import yaml
import logging
import csv
import os
import math
import pickle
from tqdm import tqdm
import src.llm_function.request_llm as request_llm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
import src.utils.util as util
import copy
import random
import base64
from datetime import datetime
import traceback

def set_logger(args, log_dir, name="active_labeling"):
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
            level=logging.INFO,
            handlers=[logging.StreamHandler()],
        )

    logger = logging.getLogger(__name__)
    for key in args:
        logger.info(f"{key}: {args[key]}")
    return logger


def get_self_generated_samples(unlabeled_set, task="crisismmd", example_num=100, language="en"):
    with open("config/prompt.yml", "r") as f:
        prompt_data = yaml.safe_load(f)
    instruction = prompt_data[task]["self_generate_samples"]
    if language == "zh":
        prompt = (
            instruction
            + f" 例如，数据集中有 {example_num} 个未标记的样本如下："
        )
    elif language == "en":
        prompt = (
            instruction
            + f" For example, {example_num} of the unlabeled samples in the dataset are as follows:"
        )
    for sample in unlabeled_set[:example_num]:
        prompt += " [" + sample.contents + "]"
    messages = [{"role": "user", "content": prompt}]
    response = request_llm.ask_open_ai(messages)
    return response.content



def save_samples(samples, out_put_path):
    with open(out_put_path, "w", newline="") as file:
        writer = csv.writer(file)
        for sample in samples:
            content = sample.contents
            label = sample.label
            writer.writerow([label, content])

def save_image_text_classification_samples(samples, out_put_path):
    with open(out_put_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "label", "text", "image_path"])
        for sample in samples:
            index = sample.index
            content = sample.contents
            label = sample.label
            image_path = sample.image_path
            writer.writerow([index, label, content, image_path])


def annotation_with_visual_llm_simple(unlabeled_set, out_put_path, task="vsnil", raw_response_path=None, image_path=None, get_cost=False):
    with open("config/prompt.yml", "r") as f:
        prompt_data = yaml.safe_load(f)
    instruction = prompt_data[task]["llm_annotation"]
    category_to_label_dict = prompt_data[task]["category_to_label"]

    with open(out_put_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "label", "text", "image_path"])

    annotated_samples = []
    llm_cost = 0
    image_data = None

    for sample in tqdm(unlabeled_set):
        text = sample.contents
        image_full_path = os.path.join(image_path, sample.image_path)
        with open(image_full_path, "rb") as f:
            image_data = f.read()
        image_data = base64.b64encode(image_data).decode("utf-8")

        if image_data is None:
            raise ValueError(f"Image data is None for sample {sample.index}")

        messsage = [
            HumanMessage(
                content=[
                    {"type": "text", "text": f"{instruction} {text} \n\nYou can refer to the image to help you make the decision."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                ]
            )
        ]

        try:
            res, cb, category, label = request_llm.ask_open_ai_with_structured_output(messsage, category_to_label_dict)
            llm_cost += cb.total_cost

            if raw_response_path:
                with open(raw_response_path, "a") as f:
                    f.write(f"{instruction} {text}" + "\n")
                    f.write(res.content + category + "\n\n")
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            print(f"label is set to -1")
            label = "-1"

        with open(out_put_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([sample.index, label, sample.contents, sample.image_path])

    if get_cost:
        return annotated_samples, llm_cost
    else:
        return annotated_samples
    

def annotation_with_visual_llm_bias(unlabeled_set, out_put_path, task="vsnil", raw_response_path=None, image_path=None, get_cost=False):
    with open("config/prompt.yml", "r") as f:
        prompt_data = yaml.safe_load(f)
    instruction = prompt_data[task]["llm_annotation_bias"]
    category_to_label_dict = prompt_data[task]["category_to_label"]

    with open(out_put_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "label", "text", "image_path"])

    annotated_samples = []
    llm_cost = 0
    image_data = None

    for sample in tqdm(unlabeled_set):
        text = sample.contents
        image_full_path = os.path.join(image_path, sample.image_path)
        with open(image_full_path, "rb") as f:
            image_data = f.read()
        image_data = base64.b64encode(image_data).decode("utf-8")

        if image_data is None:
            raise ValueError(f"Image data is None for sample {sample.index}")

        messsage = [
            HumanMessage(
                content=[
                    {"type": "text", "text": f"{instruction} {text} \n\nYou can refer to the image to help you make the decision."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                ]
            )
        ]

        try:
            res, cb, category, label = request_llm.ask_open_ai_with_structured_output(messsage, category_to_label_dict)
            llm_cost += cb.total_cost

            if raw_response_path:
                with open(raw_response_path, "a") as f:
                    f.write(f"{instruction} {text}" + "\n")
                    f.write(res.content + category + "\n\n")
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            print(f"label is set to -1")
            label = "-1"

        with open(out_put_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([sample.index, label, sample.contents, sample.image_path])

    if get_cost:
        return annotated_samples, llm_cost
    else:
        return annotated_samples
    

def annotation_with_llm_simple(unlabeled_set, out_put_path, task="vsnil", raw_response_path=None, get_cost=False):
    with open("config/prompt.yml", "r") as f:
        prompt_data = yaml.safe_load(f)
    instruction = prompt_data[task]["llm_annotation"]
    category_to_label_dict = prompt_data[task]["category_to_label"]

    with open(out_put_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "label", "text", "image_path"])

    annotated_samples = []
    llm_cost = 0

    for idx, sample in tqdm(enumerate(unlabeled_set)):
        text = sample.contents
        messsage = [HumanMessage(f"{instruction} {text}", name="example_user")]
        try:
            res, cb, category, label = request_llm.ask_open_ai_with_structured_output(messsage, category_to_label_dict)
            llm_cost += cb.total_cost
            util.append_dict_to_yml(os.path.join(os.path.dirname(out_put_path), "extra_info.yml"), {"progress": idx/len(unlabeled_set)})

            if raw_response_path:
                with open(raw_response_path, "a") as f:
                    f.write(f"{instruction} {text}" + "\n")
                    f.write(res.content + category + "\n\n")
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            label = random.choice(list(category_to_label_dict.values()))
            print(f"label is randomly set to {label}")


        with open(out_put_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([sample.index, label, sample.contents, sample.image_path])

    util.append_dict_to_yml(os.path.join(os.path.dirname(out_put_path), "extra_info.yml"), {"progress": 1})

    if get_cost:
        return annotated_samples, llm_cost
    else:
        return annotated_samples


def annotation_with_llm_swapping(unlabeled_set, out_put_path, task="vsnil", raw_response_path=None, get_cost=False):
    with open("config/prompt.yml", "r") as f:
        prompt_data = yaml.safe_load(f)
    instruction = prompt_data[task]["llm_annotation_swapping"]
    category_to_label_dict = prompt_data[task]["category_to_label"]

    with open(out_put_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "label", "text", "image_path"])

    annotated_samples = []
    llm_cost = 0

    for sample in tqdm(unlabeled_set):
        text = sample.contents
        messsage = [HumanMessage(f"{text} {instruction}", name="example_user")]
        try:
            res, cb, category, label = request_llm.ask_open_ai_with_structured_output(messsage, category_to_label_dict)
            llm_cost += cb.total_cost

            if raw_response_path:
                with open(raw_response_path, "a") as f:
                    f.write(f"{text} {instruction}" + "\n")
                    f.write(res.content + category + "\n\n")
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            print(f"label is set to -1")
            label = "-1"

        with open(out_put_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([sample.index, label, sample.contents, sample.image_path])

    if get_cost:
        return annotated_samples, llm_cost
    else:
        return annotated_samples


def annotation_with_llm_tf(unlabeled_set, out_put_path, task="crisismmd", raw_response_path=None, get_cost=False):
    with open("config/prompt.yml", "r") as f:
        prompt_data = yaml.safe_load(f)
    instruction = prompt_data[task]["llm_annotation_tf"]
    category_to_label_dict = prompt_data[task]["category_to_label"]

    with open(out_put_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "label", "text", "image_path"])


    annotated_samples = []
    llm_cost = 0

    for sample in tqdm(unlabeled_set):
        text = sample.contents
        messsage = [HumanMessage(f"{instruction} {text}", name="example_user")]
        try:
            res, cb, category, label = request_llm.ask_open_ai_with_structured_output(messsage, category_to_label_dict)
            llm_cost += cb.total_cost

            if raw_response_path:
                with open(raw_response_path, "a") as f:
                    f.write(f"{instruction} {text}" + "\n")
                    f.write(res.content + category + "\n\n")
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            print(f"label is set to -1")
            label = "-1"

        with open(out_put_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([sample.index, label, sample.contents, sample.image_path])

    if get_cost:
        return annotated_samples, llm_cost
    else:
        return annotated_samples


def annotation_with_llm_choice(unlabeled_set, out_put_path, task="vsnli", raw_response_path=None, get_cost=False):
    with open("config/prompt.yml", "r") as f:
        prompt_data = yaml.safe_load(f)
    instruction = prompt_data[task]["llm_annotation_choice"]
    category_to_label_dict = prompt_data[task]["category_to_label"]

    with open(out_put_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "label", "text", "image_path"])

    annotated_samples = []
    llm_cost = 0

    for sample in tqdm(unlabeled_set):
        text = sample.contents
        messsage = [HumanMessage(f"{instruction} {text}", name="example_user")]
        try:
            res, cb, category, label = request_llm.ask_open_ai_with_structured_output(messsage, category_to_label_dict)
            llm_cost += cb.total_cost

            if raw_response_path:
                with open(raw_response_path, "a") as f:
                    f.write(f"{instruction} {text}" + "\n")
                    f.write(res.content + category + "\n\n")
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            print(f"label is set to -1")
            label = "-1"

        with open(out_put_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([sample.index, label, sample.contents, sample.image_path])
    if get_cost:
        return annotated_samples, llm_cost
    else:
        return annotated_samples


def annotation_with_llm_bias(unlabeled_set, out_put_path, task="mr", raw_response_path=None, get_cost=False):
    with open("config/prompt.yml", "r") as f:
        prompt_data = yaml.safe_load(f)
    instruction = prompt_data[task]["llm_annotation_bias"]
    category_to_label_dict = prompt_data[task]["category_to_label"]

    with open(out_put_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "label", "text", "image_path"])

    annotated_samples = []
    llm_cost = 0

    for sample in tqdm(unlabeled_set):
        text = sample.contents
        messsage = [HumanMessage(f"{instruction} {text}", name="example_user")]
        try:
            res, cb, category, label = request_llm.ask_open_ai_with_structured_output(messsage, category_to_label_dict)
            llm_cost += cb.total_cost

            if raw_response_path:
                with open(raw_response_path, "a") as f:
                    f.write(f"{instruction} {text}" + "\n")
                    f.write(res.content + category + "\n\n")
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            print(f"label is set to -1")
            label = "-1"

        with open(out_put_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([sample.index, label, sample.contents, sample.image_path])
    if get_cost:
        return annotated_samples, llm_cost
    else:
        return annotated_samples


def get_args(task):
    with open("config/llm_config.yml", "r") as f:
        slm_config = yaml.safe_load(f)
    args = slm_config[task]
    args["OUTPUT_DIR"] = os.path.join(args["PROJECT_ROOT"], "output", task)
    return args


def llm_refine_with_confidence(task, train_most_representative_idx_path, confidence_csv_path, raw_response_path, output_file_path, log_dir, confidence_threshold=0.9, get_cost=False, visual=False, image_path=None, rules=None):
    args = get_args(task)
    logger = set_logger(args, log_dir, "llm_refine_with_confidence")

    with open(os.path.join(args["PROJECT_ROOT"], "config/prompt.yml"), "r") as f:
        prompt_data = yaml.safe_load(f)
        instruction = prompt_data[task]["llm_annotation"]
        category_to_label_dict = prompt_data[task]["category_to_label"]
        label_to_category = {v: k for k, v in category_to_label_dict.items()}

    logger.info(f"Instruction: {instruction}")

    most_representative_samples = []
    with open(train_most_representative_idx_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            most_representative_samples.append((row[0], int(row[1]), row[2], row[3]))

    logger.info(f"Most representative idx: {[sample[0] for sample in most_representative_samples]}")

    few_shot_messages = []

    if rules:
        few_shot_messages.append(
            SystemMessage(
                f"You are a expert in annotation task. Below are some suggestions for annotation process. You can refer to them to help you make the annotation.\n{rules}"
            )
        )

    for i in range(len(most_representative_samples)):
        sample = most_representative_samples[i]
        text = sample[2]
        label = sample[1]
        category = label_to_category[label]
        few_shot_messages.append(
            HumanMessage(
                f"{instruction} {text}",
                name="example_user",
            )
        )
        few_shot_messages.append(
            AIMessage(
                "",
                name="example_assistant",
                tool_calls=[{"name": "ClassificationTask", "args": {"category": category}, "id": f"{i}"}],
            )
        )
        few_shot_messages.append(
            ToolMessage(
                "",
                tool_call_id=f"{i}",
            )
        )

    with open(confidence_csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        confidence_data = [row for row in reader]

    with open(output_file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "text", "image_path"])

    to_be_annotated_samples = []
    random.seed(42)
    random.shuffle(confidence_data)
    confidence_data.sort(key=lambda x: float(x[4]))

    less_than_confidence_threthold_count = 0
    for i in range(len(confidence_data)):
        if float(confidence_data[i][4]) < confidence_threshold:
            less_than_confidence_threthold_count += 1

    if visual:
        to_be_annotated_samples_count = min(min(int(len(confidence_data) * 0.1), less_than_confidence_threthold_count),500)
    else:
        to_be_annotated_samples_count = min(less_than_confidence_threthold_count, 1000)

    extra_info_file_path = os.path.join(os.path.dirname(output_file_path), "extra_info.yml")
    util.append_dict_to_yml(extra_info_file_path, {"label_count": to_be_annotated_samples_count})
    
    for i in range(to_be_annotated_samples_count):
        to_be_annotated_samples.append(confidence_data[i])


    total_llm_cost = 0
    for idx, sample in tqdm(enumerate(to_be_annotated_samples)):
        temp_messages = copy.deepcopy(few_shot_messages)
        text = sample[2]
        if visual:
            image_full_path = os.path.join(image_path, sample[3])
            logging.info(f"image_full_path: {image_full_path}")
            with open(image_full_path, "rb") as f:
                image_data = f.read()
            image_data = base64.b64encode(image_data).decode("utf-8")
            temp_messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": f"{instruction} {text} \n\nYou can refer to the image to help you make the decision."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    ]
                )
            )
        else:
            temp_messages.append(HumanMessage(f"{instruction} {text}", name="example_user"))

        try:
            res, cb, category, label = request_llm.ask_open_ai_with_structured_output(temp_messages, category_to_label_dict)
            total_llm_cost += cb.total_cost

            if raw_response_path:
                with open(raw_response_path, "a") as f:
                    for message in temp_messages:
                        f.write(message.__class__.__name__ + ": ")
                        f.write(str(message) + "\n")
                    f.write(str(res.content) + category + "\n\n")
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
            label = random.choice(list(category_to_label_dict.values()))
            print(f"label is randomly set to {label}")

        with open(output_file_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([sample[0], label, sample[2], sample[3]])

        util.append_dict_to_yml(os.path.join(os.path.dirname(output_file_path), "extra_info.yml"), {"progress": idx/len(to_be_annotated_samples)})

    util.append_dict_to_yml(os.path.join(os.path.dirname(output_file_path), "extra_info.yml"), {"progress": 1})
    if get_cost:
        return total_llm_cost
    else:
        return None


def validate_llm_refine(logger, ground_truth_path, llm_annotation_index, entropy_data, llm_annotation_label):
    ground_truth_label = []
    with open(ground_truth_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            ground_truth_label.append(int(row[0]))
    selected_ground_truth_label = []
    for idx in llm_annotation_index:
        selected_ground_truth_label.append(ground_truth_label[idx])
    logger.info(f"Selected ground truth label: {selected_ground_truth_label}")

    previoud_label = []
    for idx in llm_annotation_index:
        previoud_label.append(int(entropy_data[idx][0]))
    logger.info(f"Previoud label: {previoud_label}")

    acc_cnt_of_previoud_label = 0
    for idx in range(len(previoud_label)):
        if previoud_label[idx] == selected_ground_truth_label[idx]:
            acc_cnt_of_previoud_label += 1

    logger.info(f"acc of previous label: {acc_cnt_of_previoud_label/len(previoud_label)}, {acc_cnt_of_previoud_label}/{len(previoud_label)}")

    acc_cnt_of_llm_annotation = 0
    for idx in range(len(llm_annotation_label)):
        if llm_annotation_label[idx] == selected_ground_truth_label[idx]:
            acc_cnt_of_llm_annotation += 1

    logger.info(f"acc of llm annotation: {acc_cnt_of_llm_annotation/len(llm_annotation_label)}, {acc_cnt_of_llm_annotation}/{len(llm_annotation_label)}")


def save_llm_refine_result(unlabeled_set, all_labels, output_file_path):
    with open(output_file_path, 'w') as f:
        writer = csv.writer(f)
        for i in range(len(unlabeled_set)):
            writer.writerow([all_labels[i]] + [unlabeled_set[i].contents] + [unlabeled_set[i].image_path])
