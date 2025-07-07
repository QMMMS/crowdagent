import src.data_processor.image_text_classification_processor_str_index as image_text_classification_processor
import src.human_function.human_annotation as human_annotation
import src.llm_function.active_labeling_seq as active_labeling
import src.slm_function.slm_fliter as slm_fliter
import src.slm_function.chinese_roberta as chinese_roberta
import src.slm_function.robust_roberta as roberta
import src.slm_function.robust_convnextv2 as convnextv2
import src.slm_function.robust_mmbt as mmbt
import src.data_function.validation as validation
import src.llm_function.request_llm as request_llm
from src.utils import util
import os
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
import csv
import yaml
import time
import logging


def llm_first_annotate(state, agent_name_list):
    loop_count = state.get("loop_count")
    if loop_count != 1:
        raise ValueError("loop_count should be 1 for the first annotation.")
    
    train_path = state.get("train_path")
    task = state.get("task")
    project_root = state.get("project_root")
    left_cnt = state.get("left_cnt")

    processor = image_text_classification_processor.ImageTextClassificationProcessor()
    unlabeled_set = processor.get_train_examples(train_path, skip_head=True)
    raw_response_path = os.path.join(project_root, "output", task, "round1", "llm_raw_output.txt")
    image_path = os.path.join(project_root, "data", task, "images")
    util.check_dir(raw_response_path)
    
    llm_total_cost = util.read_info_from_yml(os.path.join(project_root, "output", task, f"round{loop_count}", "extra_info.yml"), "budget_cost")
    llm_total_cost = 0 if llm_total_cost is None else llm_total_cost
    
    if "visual_llm_simple" in agent_name_list:
        visual_csv_path = os.path.join(project_root, "output", task, "round1", "visual_llm_simple.csv")
        if not os.path.exists(visual_csv_path):
            logging.info(f"File {visual_csv_path} does not exist. Starting annotation.")
            annotated_samples, llm_cost = active_labeling.annotation_with_visual_llm_simple(
                unlabeled_set, visual_csv_path, task=task, raw_response_path=raw_response_path, image_path=image_path, get_cost=True
            )
            llm_total_cost += llm_cost
        else:
            logging.info(f"File {visual_csv_path} already exists. Skipping annotation.")


    if "visual_llm_bias" in agent_name_list:
        visual_csv_path = os.path.join(project_root, "output", task, "round1", "visual_llm_bias.csv")
        if not os.path.exists(visual_csv_path):
            logging.info(f"File {visual_csv_path} does not exist. Starting annotation.")
            annotated_samples, llm_cost = active_labeling.annotation_with_visual_llm_bias(
                unlabeled_set, visual_csv_path, task=task, raw_response_path=raw_response_path, image_path=image_path, get_cost=True
            )
            llm_total_cost += llm_cost
        else:
            logging.info(f"File {visual_csv_path} already exists. Skipping annotation.")


    if "llm_simple" in agent_name_list:
        simple_csv_path = os.path.join(project_root, "output", task, "round1", "llm_simple.csv")
        if not os.path.exists(simple_csv_path):
            logging.info(f"File {simple_csv_path} does not exist. Starting annotation.")
            annotated_samples, llm_cost = active_labeling.annotation_with_llm_simple(
                unlabeled_set, simple_csv_path, task=task, raw_response_path=raw_response_path, get_cost=True
            )
            llm_total_cost += llm_cost
        else:
            logging.info(f"File {simple_csv_path} already exists. Skipping annotation.")


    if "llm_swapping" in agent_name_list:
        swapping_csv_path = os.path.join(project_root, "output", task, "round1", "llm_swapping.csv")
        if not os.path.exists(swapping_csv_path):
            logging.info(f"File {swapping_csv_path} does not exist. Starting annotation.")
            annotated_samples, llm_cost = active_labeling.annotation_with_llm_swapping(
                unlabeled_set, swapping_csv_path, task=task, raw_response_path=raw_response_path, get_cost=True
            )
            llm_total_cost += llm_cost
        else:
            logging.info(f"File {swapping_csv_path} already exists. Skipping annotation.")


    if "llm_tf" in agent_name_list:
        tf_csv_path = os.path.join(project_root, "output", task, "round1", "llm_tf.csv")
        if not os.path.exists(tf_csv_path):
            logging.info(f"File {tf_csv_path} does not exist. Starting annotation.")
            annotated_samples, llm_cost = active_labeling.annotation_with_llm_tf(
                unlabeled_set, tf_csv_path, task=task, raw_response_path=raw_response_path, get_cost=True
            )
            llm_total_cost += llm_cost
        else:
            logging.info(f"File {tf_csv_path} already exists. Skipping annotation.")


    if "llm_bias" in agent_name_list:
        bias_csv_path = os.path.join(project_root, "output", task, "round1", "llm_bias.csv")
        if not os.path.exists(bias_csv_path):
            logging.info(f"File {bias_csv_path} does not exist. Starting annotation.")
            annotated_samples, llm_cost = active_labeling.annotation_with_llm_bias(
                unlabeled_set, bias_csv_path, task=task, raw_response_path=raw_response_path, get_cost=True
            )
            llm_total_cost += llm_cost
        else:
            logging.info(f"File {bias_csv_path} already exists. Skipping annotation.")


    if "llm_choice" in agent_name_list:
        choice_csv_path = os.path.join(project_root, "output", task, "round1", "llm_choice.csv")
        if not os.path.exists(choice_csv_path):
            logging.info(f"File {choice_csv_path} does not exist. Starting annotation.")
            annotated_samples, llm_cost = active_labeling.annotation_with_llm_choice(
                unlabeled_set, choice_csv_path, task=task, raw_response_path=raw_response_path, get_cost=True
            )
            llm_total_cost += llm_cost
        else:
            logging.info(f"File {choice_csv_path} already exists. Skipping annotation.")

    extra_info_path = os.path.join(project_root, "output", task, "round1", "extra_info.yml")
    util.append_dict_to_yml(extra_info_path, {"budget_cost": llm_total_cost})



def human_annotate(state, online=False):
    loop_count = state.get("loop_count")
    confidence_threshold = state.get("confidence_threshold")
    if loop_count == 1:
        raise ValueError("first annotation should be done by LLM.")
    
    task = state.get("task")
    project_root = state.get("project_root")
    last_data_embeddings_path = state.get("last_data_embeddings_path")
    if os.path.exists(last_data_embeddings_path):
        mode = "coreset"
    else:
        mode = "confidence"
    
    if not online:
        unlabeled_gt_file_path = os.path.join(project_root, "data", task, "unlabeled_gt.csv")

    labeled_gt_file_path = state.get("dev_path")

    last_round_aggregated_file_path = os.path.join(
        state.get("project_root"), "output", state.get("task"), f"round{loop_count-1}", "train_aggregated.csv"
    )
    output_file_path = os.path.join(
        state.get("project_root"), "output", state.get("task"), f"round{loop_count}", "human.csv"
    )

    to_be_annotated_file_path = os.path.join(
        state.get("project_root"), "output", state.get("task"), f"round{loop_count}", "to_be_annotated.csv"
    )


    util.check_dir(output_file_path)
    annotation_rate = 0.05

    util.append_dict_to_yml(os.path.join(os.path.dirname(output_file_path), "extra_info.yml"), {"agent_name": "human"})

    if not os.path.exists(output_file_path):
        logging.info(f"File {output_file_path} does not exist. Starting human annotation.")
        if not online:
            if not os.path.exists(last_data_embeddings_path):
                human_annotation.human_annotation_with_confidence(last_round_aggregated_file_path, output_file_path, unlabeled_gt_file_path, labeled_gt_file_path, confidence_threshold=confidence_threshold, annotation_rate=annotation_rate)
            else:
                human_annotation.human_annotation_with_coreset(last_round_aggregated_file_path, output_file_path, unlabeled_gt_file_path, labeled_gt_file_path, last_data_embeddings_path, confidence_threshold=confidence_threshold, annotation_rate=annotation_rate)
        else:
            if not os.path.exists(last_data_embeddings_path):
                human_annotation.human_annotation_with_confidence_online(last_round_aggregated_file_path, to_be_annotated_file_path, confidence_threshold=confidence_threshold, annotation_rate=annotation_rate)
            else:
                human_annotation.human_annotation_with_coreset_online(last_round_aggregated_file_path, to_be_annotated_file_path, last_data_embeddings_path, confidence_threshold=confidence_threshold, annotation_rate=annotation_rate)

            while True:
                if not os.path.exists(output_file_path):
                    logging.info(f"Waiting for human annotation to complete...")
                    time.sleep(10)
                else:
                    logging.info(f"Human annotation completed. Output file created: {output_file_path}")
                    break
    else:
        logging.info(f"File {output_file_path} already exists. Skipping human annotation.")

    return mode


async def youling_annotate(state):
    logging.info(f"Youling annotate")
    loop_count = state.get("loop_count")
    confidence_threshold = state.get("confidence_threshold")
    if loop_count == 1:
        raise ValueError("first annotation should be done by LLM.")
    
    task = state.get("task")
    project_root = state.get("project_root")
    img_base_path = os.path.join(project_root, "data", task, "images")
    last_round_aggregated_file_path = os.path.join(
        state.get("project_root"), "output", state.get("task"), f"round{loop_count-1}", "train_aggregated.csv"
    )
    output_file_path = os.path.join(
        state.get("project_root"), "output", state.get("task"), f"round{loop_count}", "human.csv"
    )

    util.check_dir(output_file_path)

    if not os.path.exists(output_file_path):
        logging.info(f"File {output_file_path} does not exist. Starting human annotation.")
        await human_annotation.human_annotation_with_youling(last_round_aggregated_file_path, output_file_path, img_base_path, confidence_threshold=confidence_threshold, annotation_rate=0.05)
    else:
        logging.info(f"File {output_file_path} already exists. Skipping human annotation.")


def chinese_roberta_train(state):
    loop_count = state.get("loop_count")
    if loop_count == 1:
        raise ValueError("first annotation should be done by LLM.")
    
    task = state.get("task")
    project_root = state.get("project_root")
    dev_file_path = state.get("dev_path")
    confidence_threshold = state.get("confidence_threshold")
    last_round_train_path = os.path.join(project_root, "output", task, f"round{loop_count-1}", "train_aggregated.csv")
    model_save_path = os.path.join(project_root, "output", task, f"round{loop_count}")
    log_dir = os.path.join(project_root, "output", task, f"round{loop_count}")
    model_checkpoint_save_path = os.path.join(model_save_path, "checkpoint-best-val")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(model_checkpoint_save_path):
        logging.info(f"Dir {model_checkpoint_save_path} does not exist. Starting Chinese RoBERTa training.")

        util.write_info_to_yml(
            os.path.join(log_dir, "extra_info.yml"),
            "start_time",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        chinese_roberta.train(
            task=task,
            train_file_path=last_round_train_path,
            dev_file_path=dev_file_path,
            model_save_path=model_save_path,
            log_dir=log_dir,
            confidence_threshold=confidence_threshold,
        )

        util.write_info_to_yml(
            os.path.join(log_dir, "extra_info.yml"),
            "end_time",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    else:
        logging.info(f"Dir {model_checkpoint_save_path} already exists. Skipping Chinese RoBERTa training.")


def roberta_train(state):
    loop_count = state.get("loop_count")
    if loop_count == 1:
        raise ValueError("first annotation should be done by LLM.")
    
    task = state.get("task")
    project_root = state.get("project_root")
    dev_file_path = state.get("dev_path")
    confidence_threshold = state.get("confidence_threshold")
    last_round_train_path = os.path.join(project_root, "output", task, f"round{loop_count-1}", "train_aggregated.csv")
    model_save_path = os.path.join(project_root, "output", task, f"round{loop_count}")
    log_dir = os.path.join(project_root, "output", task, f"round{loop_count}")
    model_checkpoint_save_path = os.path.join(model_save_path, "checkpoint-best-val")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(model_checkpoint_save_path):
        logging.info(f"Dir {model_checkpoint_save_path} does not exist. Starting Roberta training.")    

        util.write_info_to_yml(
            os.path.join(log_dir, "extra_info.yml"),
            "start_time",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        roberta.train(
            task=task,
            train_file_path=last_round_train_path,
            dev_file_path=dev_file_path,
            model_save_path=model_save_path,
            log_dir=log_dir,
            confidence_threshold=confidence_threshold,
        )

        util.write_info_to_yml(
            os.path.join(log_dir, "extra_info.yml"),
            "end_time",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    else:
        logging.info(f"Dir {model_checkpoint_save_path} already exists. Skipping Roberta training.")
        


def slm_fliter_samples(state, type, clean_rate=0.2):
    loop_count = state.get("loop_count")
    if loop_count == 1:
        raise ValueError("first annotation should be done by LLM.")
    
    task = state.get("task")
    project_root = state.get("project_root")
    confidence_threshold = state.get("confidence_threshold")
    
    model_path = os.path.join(project_root, "output", task, f"round{loop_count}", "checkpoint-best-val")
    confidence_csv_path = os.path.join(project_root, "output", task, f"round{loop_count-1}", "train_aggregated.csv")
    log_dir = os.path.join(project_root, "output", task, f"round{loop_count}")
    confidence_csv_out_path = os.path.join(project_root, "output", task, f"round{loop_count}", f"{type}_high_confidence.csv")
    other_confidence_csv_out_path = os.path.join(project_root, "output", task, f"round{loop_count}", f"{type}_other_confidence.csv")
    most_representative_idx_output_file_path = os.path.join(project_root, "output", task, f"round{loop_count}", "most_representative_idx.csv")
    data_embeddings_path = os.path.join(project_root, "output", task, f"round{loop_count}", "data_embeddings.pt")

    if not os.path.exists(confidence_csv_out_path):
        logging.info(f"File {confidence_csv_out_path} does not exist. Starting filtering.")
        util.check_dir(confidence_csv_out_path)

        try:
            slm_fliter.fliter(task=task,
                model_path=model_path,
                confidence_csv_path=confidence_csv_path,
                log_dir=log_dir,
                confidence_csv_out_path=confidence_csv_out_path,
                other_confidence_csv_out_path=other_confidence_csv_out_path,
                most_representative_idx_output_file_path=most_representative_idx_output_file_path,
                type=type,
                confidence_threshold=confidence_threshold,
                clean_rate=clean_rate,
                data_embeddings_path=data_embeddings_path,
            )
        except Exception as e:
            import traceback
            logging.error(f"Error: {e}")
            logging.error(traceback.format_exc())
            raise e
    else:
        logging.info(f"File {confidence_csv_out_path} already exists. Skipping filtering.")

    return most_representative_idx_output_file_path, data_embeddings_path


def convnextv2_train(state):
    loop_count = state.get("loop_count")
    if loop_count == 1:
        raise ValueError("first annotation should be done by LLM.")
    
    task = state.get("task")
    project_root = state.get("project_root")
    dev_file_path = state.get("dev_path")
    confidence_threshold = state.get("confidence_threshold")
    last_round_train_path = os.path.join(project_root, "output", task, f"round{loop_count-1}", "train_aggregated.csv")
    model_save_path = os.path.join(project_root, "output", task, f"round{loop_count}")
    log_dir = os.path.join(project_root, "output", task, f"round{loop_count}")
    model_checkpoint_save_path = os.path.join(model_save_path, "checkpoint-best-val")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(model_checkpoint_save_path):
        logging.info(f"Dir {model_checkpoint_save_path} does not exist. Starting ConvNextV2 training.")

        util.write_info_to_yml(
            os.path.join(log_dir, "extra_info.yml"),
            "start_time",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        convnextv2.train(
            task=task,
            train_file_path=last_round_train_path,
            dev_file_path=dev_file_path,
            model_save_path=model_save_path,
            log_dir=log_dir,
            confidence_threshold=confidence_threshold,
        )

        util.write_info_to_yml(
            os.path.join(log_dir, "extra_info.yml"),
            "end_time",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    else:
        logging.info(f"Dir {model_checkpoint_save_path} already exists. Skipping ConvNextV2 training.")


def slm_train(state, type):
    loop_count = state.get("loop_count")
    id_to_label = state.get("id_to_label")
    if loop_count == 1:
        raise ValueError("first annotation should be done by LLM.")
    
    task = state.get("task")
    project_root = state.get("project_root")
    dev_file_path = state.get("dev_path")
    confidence_threshold = state.get("confidence_threshold")
    last_round_train_path = os.path.join(project_root, "output", task, f"round{loop_count-1}", "train_aggregated.csv")
    model_save_path = os.path.join(project_root, "output", task, f"round{loop_count}")
    log_dir = os.path.join(project_root, "output", task, f"round{loop_count}")
    model_checkpoint_save_path = os.path.join(model_save_path, "checkpoint-best-val")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(model_checkpoint_save_path):
        logging.info(f"Dir {model_checkpoint_save_path} does not exist. Starting {type} training.")

        util.write_info_to_yml(
            os.path.join(log_dir, "extra_info.yml"),
            "start_time",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        if type == "convnextv2":
            convnextv2.train(
                task=task,
                train_file_path=last_round_train_path,
                dev_file_path=dev_file_path,
                model_save_path=model_save_path,
                log_dir=log_dir,
                confidence_threshold=confidence_threshold,
                id_to_label=id_to_label,
            )
        elif type == "mmbt":
            try:
                mmbt.train(
                    task=task,
                    train_file_path=last_round_train_path,
                    dev_file_path=dev_file_path,
                    model_save_path=model_save_path,
                    log_dir=log_dir,
                    confidence_threshold=confidence_threshold,
                    num_labels=len(id_to_label),
                )
            except Exception as e:
                import traceback
                logging.error(f"Error: {e}")
                logging.error(traceback.format_exc())
                raise e
        elif type == "roberta":
            roberta.train(
                task=task,
                train_file_path=last_round_train_path,
                dev_file_path=dev_file_path,
                model_save_path=model_save_path,
                log_dir=log_dir,
                confidence_threshold=confidence_threshold,
                num_labels=len(id_to_label),
            )
        elif type == "chinese_roberta":
            chinese_roberta.train(
                task=task,
                train_file_path=last_round_train_path,
                dev_file_path=dev_file_path,
                model_save_path=model_save_path,
                log_dir=log_dir,
                confidence_threshold=confidence_threshold,
            )
        else:
            raise ValueError(f"Invalid type: {type}. Supported types are: convnextv2, mmbt, roberta, chinese_roberta.")

        util.write_info_to_yml(
            os.path.join(log_dir, "extra_info.yml"),
            "end_time",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    else:
        logging.info(f"Dir {model_checkpoint_save_path} already exists. Skipping {type} training.")


def llm_refine_annotate(state, visual=False):
    loop_count = state.get("loop_count")
    confidence_threshold = state.get("confidence_threshold")
    if loop_count == 1:
        raise ValueError("first annotation should be done by LLM.")
    
    train_most_representative_idx_path = state.get("last_selected_samples_path")
    if not train_most_representative_idx_path:
        raise ValueError("last_selected_samples_path should be provided.")
    
    loop_count = state.get("loop_count")
    task = state.get("task")
    project_root = state.get("project_root")
    rules_path = state.get("rules_path")
    prefix = "text" if not visual else "visual"
    confidence_csv_path = os.path.join(project_root, "output", task, f"round{loop_count-1}", "train_aggregated.csv")
    raw_response_path = os.path.join(project_root, "output", task, f"round{loop_count}", f"{prefix}_llm_refine_raw_output.txt")
    output_file_path = os.path.join(project_root, "output", task, f"round{loop_count}", f"{prefix}_llm_refine.csv")
    image_path = os.path.join(project_root, "data", task, "images")
    log_dir = os.path.join(project_root, "output", task, f"round{loop_count}")
    util.check_dir(output_file_path)

    extra_info_file_path = os.path.join(project_root, "output", task, f"round{loop_count}", "extra_info.yml")
    util.append_dict_to_yml(extra_info_file_path, {"agent_name": "gpt-4o-mini(visual)" if visual else "gpt-4o-mini"})

    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = f.read()
    else:
        rules = ""


    if not os.path.exists(output_file_path):
        logging.info(f"File {output_file_path} does not exist. Starting LLM refine annotation.")

        try:
            llm_cost = active_labeling.llm_refine_with_confidence(
                task,
                train_most_representative_idx_path,
                confidence_csv_path,
                raw_response_path,
                output_file_path,
                log_dir,
                confidence_threshold=confidence_threshold,
                get_cost=True,
                visual=visual,
                image_path=image_path,
                rules=rules
            )
        except Exception as e:
            import traceback
            logging.error(f"Error: {e}")
            logging.error(traceback.format_exc())
            raise e

        util.write_info_to_yml(
            os.path.join(log_dir, "extra_info.yml"),
            "budget_cost",
            llm_cost
        )

    else:
        logging.info(f"File {output_file_path} already exists. Skipping LLM refine annotation.")


def get_cost(state):
    cost_role = state.get("cost_role")

    cost_dict = {
        "gpt-4o-mini": (0.6, 2.4),
        "human": 0.015, # $ per sample
        "1080ti": 0.1,  # $ per hour
    }

    if cost_role not in cost_dict:
        raise ValueError(f"Invalid type: {cost_role}. Supported types are: {', '.join(cost_dict.keys())}")
    
    loop_count = state.get("loop_count")
    project_root = state.get("project_root")
    task = state.get("task")

    if cost_role == "1080ti":
        time_cost = util.calculate_time_cost(os.path.join(project_root, "output", task, f"round{loop_count}", "extra_info.yml"))
        return cost_dict[cost_role] * time_cost
    
    if cost_role == "gpt-4o-mini":
        extra_info_path = os.path.join(project_root, "output", task, f"round{loop_count}", "extra_info.yml")
        if not os.path.exists(extra_info_path):
            raise ValueError(f"File {extra_info_path} does not exist. Please run the script again.")
        return util.read_info_from_yml(extra_info_path, "budget_cost")
    
    if cost_role == "human":
        output_file_path = os.path.join(project_root, "output", task, f"round{loop_count}", "human.csv")
        with open(output_file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            data = [row for row in reader]
        return len(data) * cost_dict[cost_role]

    return cost_dict[cost_role]


def get_confidence_distribution(state):
    loop_count = state.get("loop_count")
    task = state.get("task")
    project_root = state.get("project_root")
    confidence_threshold = state.get("confidence_threshold")
    confidence_log_file_path = os.path.join(project_root, "output", task, f"round{loop_count}", "train_aggregated.csv")
    count_dict, average_confidence, count_confidence_under_threshold = util.display_confidence_distribution(confidence_log_file_path, confidence_threshold)
    summary = ""
    summary += f"Number of samples with very low confidence (0 ~ 0.3): {count_dict['0~0.3']}\n"
    summary += f"Number of samples with low confidence (0.3 ~ 0.6): {count_dict['0.3~0.6']}\n"
    summary += f"Number of samples with medium confidence (0.6 ~ 0.9): {count_dict['0.6~0.9']}\n"
    summary += f"Number of samples with high confidence (0.9 ~ 1): {count_dict['0.9~1']}\n"
    summary += f"Average confidence: {average_confidence}\n"
    summary += f"There are {count_confidence_under_threshold} samples with confidence below {confidence_threshold}, which need to be annotated again."
    return summary, count_confidence_under_threshold


def get_planner_instruction():
    message = """You are a **Scheduling Agent**, responsible for selecting the appropriate annotator in the annotation task. The goal is to improve the confidence of each sample in the dataset to 0.99 or more, and control the cost. For each round, you can refer to the annotators' profile to help you select the appropriate annotator. You must first state your reasoning, then declare your choice of annotator. Your reasoning should be based on the following principles: 
1) **Justify Your Choice**: Explain your selection. Your analysis can be based on "current state analysis", "historical annotator feedback", and "next round annotator selection".
2) **Diversify Annotators**: Do not use the same annotator or modality consecutively. For multi-modal tasks, check the calling path and actively alternate between text and vision models for multimodal tasks, and between LLMs and SLMs, to leverage their unique strengths.
3) **Utilize Human Experts Strategically**: Human annotation is a high-cost, high-quality resource for resolving ambiguity. Use it sparingly, considering it when model performance stagnates. A general guideline is to request human input once every five rounds.
4) **Integrate All Feedback**: While you have the final authority, your decision should be informed by the analysis provided by the Quality Assurance (QA) and Financing Agents.
5) **Iterative Learning**: Remember that all annotators in subsequent rounds will learn from the results of the current round to improve overall accuracy. 
Only when the confidence of all samples is 0.99 or more can the annotation task be considered complete. Before that, each round must select an annotator for annotation.
""" 
    return message


def get_quality_review_instruction(confidence_distribution, count_labeled_samples_and_acc, quality_chain_info):
    message = """You are a **Quality Assurance Agent**, you are responsible for auditing the quality of the annotation process after each round. Your goal is to evaluate performance, identify error patterns, and provide data-driven insights and recommendations to guide the subsequent annotation rounds. For each round, structure your analysis according to the following directives:
1) **Overall Performance Audit**: Evaluate the overall dataset's current metrics, including the confidence distribution, average confidence, and cumulative accuracy of all samples.
2) **Current Round Error Analysis**: Analyze the confidence and accuracy of the specific samples annotated in this round. By cross-referencing with historical data, identify any newly introduced errors and summarize their potential causes.
3) **Historical Annotator Comparison**: Compare the historical effectiveness of different annotators based on their calling paths. When conducting this analysis, account for the fact that sample difficulty typically increases in later rounds. Based on this, provide a recommendation regarding annotator diversification.
4) **Guidance on Human Intervention**: Provide a specific recommendation on the use of human annotators. Given their high cost, advise deploying them strategically, primarily when machine-driven accuracy has stagnated. A general heuristic is to consider human intervention approximately every five rounds.
5) **Output \& Format**: Deliver your analysis directly, without introducing your role. The report should be a concise text of approximately 300 words. Your findings and recommendations will be used by the Scheduling Agent to plan the next round.
6) **Basic Information**: """
    message += f"\n{confidence_distribution}\n{count_labeled_samples_and_acc}\n{quality_chain_info}"
    return message


def get_sample_check_instruction():
    message = """You are a quality inspector, in the annotation task process, after each round of annotation, you are responsible for checking the wrong samples and labels produced by the annotator. You need to summarize the possible causes of the error and the methods to avoid the error, and help the annotator set up annotation rules through the form of annotation rules. Task requirements and suggestions:
1. Observe the wrong annotation sample examples and think about the possible causes of the error.
2. Based on the possible causes of the error, think about the methods to avoid the error.
3. Summarize your analysis content in 300 words or less, and then give detailed annotation rules through the tool to pass your annotation rules. The annotation rules have no word limit.
4. When analyzing, you need to give specific examples for specific samples, avoid general discussion. Similarly, when giving annotation rules, you also need to give specific examples for specific samples, avoid general discussion.
5. Later, the annotator will see your suggestions and annotation rules in the next round of tasks.
6. Answer without repeating your identity. First give a 300-word analysis output, and then you should **call the tool** to save your annotation rules. Do not directly output the annotation rules with json in the answer, but use the tool to output the annotation rules.
7. The wrong annotation samples for this round are as follows:"""
    return message


def get_financial_review_instruction(cost, now_budget, financial_chain_info):
    message = """You are a **Financing Agent**, your primary function is to serve as the chief financial analyst for this annotation project. Your goal is to monitor the project's financial health, analyze the cost-effectiveness of the annotation strategy, and provide data-driven advice to ensure the project meets its quality targets within the allocated budget. For each round, conduct your analysis based on the following principles: 
 1) **Financial \& Performance Review**: Synthesize budget cost with the quality report from the QA Agent to conduct a comprehensive cost-effectiveness analysis for the current round. Review the historical performance and calling paths of all annotators to compare their long-term cost-performance ratios. 
 2) **Strategic Cost-Management Recommendations**: Based on your analysis, provide actionable suggestions for future rounds. If cost-effectiveness is low or budget consumption is unreasonable, explicitly state your concerns and recommend corrective actions. Advise on annotator diversification. Acknowledge the different pricing models (e.g., per-token for LLMs, per-sample for humans, per-hour for SLMs) and recommend against consecutively using the same annotator. 
 3) **Human Annotation Advisory**: Treat human annotation as a high-cost, high-impact resource. Advise caution in its deployment. Recommend deploying human experts only when necessary, for instance, when machine-only rounds show stagnating accuracy. A general heuristic is to suggest human intervention approximately every five rounds, but this should be adapted based on the current performance trend and budget runway. 
 4) **Output \& Format**: Begin your analysis directly without introducing your identity. Structure your response as a concise financial report. Aim for a text output of approximately 300 words. Your suggestions will be reviewed by the Scheduling Agent for the next round. 
 5) **Basic Information**: """
    message += f"\nThe cost of this round is {cost} $, and the remaining budget is {now_budget} $.\n{financial_chain_info}"
    return message


def get_user_profile_instruction():
    message = """You are an annotation quality analyst responsible for evaluating annotator capabilities through confusion matrix. Follow these steps precisely:
1. Analyze confusion matrix thoroughly. Examine percentage distribution of each "true label-predicted label" combination. Focus on diagonal values (correct annotations) and off-diagonal patterns (error types).
2. Assess category-specific performance. Identify confusion patterns: Which categories are frequently mixed? Where do errors flow? Determine strength gaps: Categories significantly above/below average accuracy.
3. Generate capability profile. Identify strengths and pinpoint weaknesses. Analyze bias: Explain why specific categories are consistently mislabeled.
4. Answer without repeating your identity. First give a 300-word analysis output, and then you should **call the tool** to save your capability profile. Do not directly output the capability profile with json in the answer!
Base your analysis on this confusion matrix:"""
    return message


def get_human_request(task, project_root):
    with open(os.path.join(project_root, "config", "prompt.yml"), "r") as f:
        extra_info = yaml.load(f, Loader=yaml.FullLoader)
    return extra_info[task]["human_request"]


def get_annotation_agents(task, project_root):
    with open(os.path.join(project_root, "config", "prompt.yml"), "r") as f:
        extra_info = yaml.load(f, Loader=yaml.FullLoader)
    return extra_info[task]["annotation_agent"]


def get_budget(task, project_root):
    with open(os.path.join(project_root, "config", "llm_config.yml"), "r") as f:
        extra_info = yaml.load(f, Loader=yaml.FullLoader)
    return extra_info[task]["BUDGET"]


def get_left_cnt(train_path):
    with open(train_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        data = [row for row in reader]
    return len(data)


def get_detail_labels(task, project_root):
    with open(os.path.join(project_root, "config", "llm_config.yml"), "r") as f:
        extra_info = yaml.load(f, Loader=yaml.FullLoader)
    label_to_id = extra_info[task]["DETAIL_LABELS"]
    id_to_label = {v: k for k, v in label_to_id.items()}
    return id_to_label


def update_quality_chain_info(state, acc, left_cnt):
    chain_role = state.get("chain_role")
    quality_chain_info = state.get("quality_chain_info")
    quality_chain_info += f"({chain_role},{acc:.4f},{left_cnt}) =>"
    return quality_chain_info


def update_financial_chain_info(state, cost, now_budget):
    chain_role = state.get("chain_role")
    financial_chain_info = state.get("financial_chain_info")
    financial_chain_info += f"({chain_role},{cost:.2f},{now_budget:.2f}) =>"
    return financial_chain_info

def update_extra_info(state, dictionary):
    round_count = state.get("loop_count")
    task = state.get("task")
    project_root = state.get("project_root")
    extra_info_file_path = os.path.join(project_root, "output", task, f"round{round_count}", "extra_info.yml")
    if not os.path.exists(os.path.dirname(extra_info_file_path)):
        os.makedirs(os.path.dirname(extra_info_file_path))
    if not os.path.exists(extra_info_file_path):
        with open(extra_info_file_path, "w", encoding="utf-8") as f:
            f.write("{}")
    util.append_dict_to_yml(extra_info_file_path, dictionary)

def get_wrong_samples(state):

    loop_count = state.get("loop_count")
    task = state.get("task")
    project_root = state.get("project_root")
    dev_file_path = state.get("dev_path")
    agent_name_list = state.get("agent_name_list")
    role = agent_name_list[0]
    id_to_label = state.get("id_to_label")
    
    part_annotataion_path = os.path.join(project_root, "output", task, f"round{loop_count}", f"{role}.csv")
    wrong_samples = util.get_wrong_samples(part_annotataion_path, dev_file_path)

    ret = ""
    for wrong_sample in wrong_samples[:50]:
        idx, text, pred_label, gt_label = wrong_sample
        ret += f"Sample{idx}, Text: {text}, Predict Label: {id_to_label[int(pred_label)]}, True Label:{id_to_label[int(gt_label)]}。\n"

    return ret


def get_wrong_samples_analysis(state, wrong_samples):
    loop_count = state.get("loop_count")
    task = state.get("task")
    project_root = state.get("project_root")

    rules_file_path = os.path.join(project_root, "output", task, f"round{loop_count}", f"annotation_rules.txt")

    sample_check_instruction = get_sample_check_instruction()
    messages = [
        SystemMessage(content=sample_check_instruction),
        HumanMessage(content=wrong_samples+"\n\nFirst give a 300-word analysis output of the wrong samples, then you must call the SaveAnnotationRules tool to save your annotation rules. Do not directly output the annotation rules with json in the answer!")
    ]
    res, cb, rules = request_llm.ask_open_ai_wrong_samples_review(messages)
    with open(rules_file_path, "w", encoding="utf-8") as f:
        f.write(rules)
    return res.content, rules_file_path, res


def get_user_profile(state, confusion_matrix_readable):
    loop_count = state.get("loop_count")
    task = state.get("task")
    project_root = state.get("project_root")
    task_extra_info_file_path = os.path.join(project_root, "output", task, "task_extra_info.yml")
    profile_role = state.get("profile_role")

    user_profile_instruction = get_user_profile_instruction()
    messages = [
        SystemMessage(content=user_profile_instruction),
        HumanMessage(content=confusion_matrix_readable+"\n\nFirst give a 300-word analysis, then you must call the SaveCapabilityProfile tool to save your generated capability profile. Do not directly output the capability profile with json in the answer!")
    ]
    res, cb, profile = request_llm.ask_open_ai_generate_user_profile(messages)
    util.append_dict_to_yml(task_extra_info_file_path, {f"{profile_role}": profile})
    return res.content, profile, res



def evaluate_on_labeled_samples(state):
    loop_count = state.get("loop_count")
    task = state.get("task")
    project_root = state.get("project_root")
    dev_file_path = state.get("dev_path")
    role = state.get("evaluate_role")
    acc, cnt = validation.get_val_acc(
        os.path.join(project_root, "output", task, f"round{loop_count}", "train_aggregated.csv"),
        dev_file_path
    )

    if loop_count == 1:
        if "llm" not in role:
            raise ValueError("first annotation should be done by LLM.")
        return f"The initial labeling processed all samples, with an estimated accuracy of {acc}.", acc
    
    if role == "llm" or role == "human" or "llm_refine" in role:
        last_aggregated_path = os.path.join(project_root, "output", task, f"round{loop_count-1}", "train_aggregated.csv")
        this_aggregated_path = os.path.join(project_root, "output", task, f"round{loop_count}", "train_aggregated.csv")
        part_annotataion_path = os.path.join(project_root, "output", task, f"round{loop_count}", f"{role}.csv")
        part_cnt, last_acc, part_acc, this_acc = util.evaluate_selected_samples(
            part_annotataion_path, last_aggregated_path, this_aggregated_path, dev_file_path
        )

        message = f"This round selected to label {part_cnt} samples.\nOn these {part_cnt} samples, the estimated accuracy of the last round is {last_acc}, the estimated accuracy of this round is {part_acc}, and the estimated accuracy of the aggregated label is {this_acc}."
        message += f"\nThe estimated accuracy of all samples is {acc}."
        return message, acc
    
    high_confidence_path = os.path.join(project_root, "output", task, f"round{loop_count}", f"{role}_high_confidence.csv")
    other_confidence_path = os.path.join(project_root, "output", task, f"round{loop_count}", f"{role}_other_confidence.csv")
    last_aggregated_path = os.path.join(project_root, "output", task, f"round{loop_count-1}", "train_aggregated.csv")
    this_aggregated_path = os.path.join(project_root, "output", task, f"round{loop_count}", "train_aggregated.csv")

    high_conf_cnt, last_acc, part_acc, this_acc, last_average_conf, this_average_conf = util.evaluate_selected_samples(
        high_confidence_path, last_aggregated_path, this_aggregated_path, dev_file_path, get_ave_conf=True
    )
    message = f"\nThe small model selected to label {high_conf_cnt} high-confidence samples.\nOn these {high_conf_cnt} samples, the estimated accuracy is {part_acc}, the average confidence from {last_average_conf} to {this_average_conf}."

    other_cnt, last_acc, part_acc, this_acc = util.evaluate_selected_samples(
        other_confidence_path, last_aggregated_path, this_aggregated_path, dev_file_path
    )
    message += f"\nIn addition, the small model labeled {other_cnt} other samples.\nOn these {other_cnt} samples, the estimated accuracy of the last round is {last_acc}, the estimated accuracy of this round is {part_acc}, and the estimated accuracy of the aggregated label is {this_acc}."
    message += f"\nThe estimated accuracy of all samples is {acc}."
    return message, acc


def get_confusion_matrix_on_labeled_samples(state):
    loop_count = state.get("loop_count")
    task = state.get("task")
    project_root = state.get("project_root")
    dev_file_path = state.get("dev_path")
    agent_name_list = state.get("agent_name_list")
    id_to_label = state.get("id_to_label")

    for agent_name in agent_name_list:

        if task == "conll":
            validation.val_output_confusion_matrix_ner(
                os.path.join(project_root, "output", task, f"round{loop_count}", f"{agent_name}.csv"),
                dev_file_path,
                f"{loop_count}_{agent_name}",
                os.path.join(project_root, "output", task, "confusion_matrix.yml"),
                num_labels=len(id_to_label)
            )

        else:
            validation.val_output_confusion_matrix(
                os.path.join(project_root, "output", task, f"round{loop_count}", f"{agent_name}.csv"),
                dev_file_path,
                f"{loop_count}_{agent_name}",
                os.path.join(project_root, "output", task, "confusion_matrix.yml"),
                num_labels=len(id_to_label)
            )


def tag_aggregation(state):
    loop_count = state.get("loop_count")
    task = state.get("task")
    project_root = state.get("project_root")
    train_file_path = state.get("train_path")
    id_to_label = state.get("id_to_label")

    if task == "conll":
        validation.tag_aggregation_bayes_ner(
            os.path.join(project_root, "output", task),
            os.path.join(project_root, "output", task, f"round{loop_count}", "train_aggregated.csv"),
            train_file_path,
            round_limit=loop_count,
            num_labels=len(id_to_label)
        )

    else:
        validation.tag_aggregation_bayes(
            os.path.join(project_root, "output", task),
            os.path.join(project_root, "output", task, f"round{loop_count}", "train_aggregated.csv"),
            train_file_path,
            round_limit=loop_count,
            num_labels=len(id_to_label)
        )


def save_chunk(chunk, task, project_root):
    yml_path = os.path.join(project_root, "output", task, "task_extra_info.yml")
    dict_data = {"chunk": str(chunk)}
    util.append_dict_to_yml(yml_path, dict_data)


def update_status(task, project_root, status):
    yml_path = os.path.join(project_root, "output", task, "task_extra_info.yml")
    dict_data = {"status": status}
    util.append_dict_to_yml(yml_path, dict_data)


def read_confusion_matrix(state):
    loop_count = state.get("loop_count")
    task = state.get("task")
    project_root = state.get("project_root")
    task_dir = os.path.join(project_root, "output", task)
    confusion_matrix = util.read_confusion_matrix(task_dir, loop_count)
    id_to_label = state.get("id_to_label")

    for key in confusion_matrix.keys():
        matrix = confusion_matrix[key]
        break

    matrix_readble = ""
    for key in matrix.keys():
        if "total_cnt" in key:
            continue
        w, t = key.split("_")[1], key.split("_")[3]
        matrix_readble += f"True label: {id_to_label[int(t)]}, Predicted label: {id_to_label[int(w)]}, Count: {matrix[key]}\n"
    return matrix_readble


def get_annotator_profiles(state):
    task = state.get("task")
    project_root = state.get("project_root")
    profile_text = "Annotator Profiles:\n\n"
    task_extra_info_file_path = os.path.join(project_root, "output", task, "task_extra_info.yml")
    keys = ["vlm_annotate_tool", "visual_slm_annotate_tool", "text_slm_annotate_tool", "multi_modal_slm_annotate_tool", "llm_annotate_tool", "human_annotate_tool"]

    for key in keys:
        profile = util.read_info_from_yml(task_extra_info_file_path, key)
        if profile:
            profile_text += f"{key}:\n{profile}\n\n"
        else:
            profile_text += f"{key}: No profile available.\n\n"

    return profile_text

