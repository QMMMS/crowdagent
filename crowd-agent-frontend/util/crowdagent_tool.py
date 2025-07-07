from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
import pandas as pd
import streamlit as st
import numpy as np
import os
import yaml
from streamlit_echarts import st_echarts
import csv
import plotly.express as px
import shutil
import time
import util.read_info as crowdagent_tool
import math
import base64
import copy
from util.path_encryption import simple_encrypt, simple_decrypt, encrypt_path, decrypt_path


def read_agents_messages(backend_path, tesk):
    try:
        with open(f"{backend_path}/output/{tesk}/task_extra_info.yml", "r", encoding="utf8") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        str_messages = data["chunk"]
        dict_messages = eval(str_messages)
        return dict_messages["messages"]
    except Exception as e:
        st.info(f"Agents messages not available yet, please wait...")
        st.info(f"Error: {e}")
        return None

def get_task_status(backend_path, task_name):
    try:
        with open(f"{backend_path}/output/{task_name}/task_extra_info.yml", "r", encoding="utf8") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return data["status"]
    except Exception as e:
        return "Running"


def get_agent_name_list(agents_messages, status="Running"):
    agent_name_list = []
    for message in agents_messages[1:]:
        if isinstance(message, AIMessage):
            content = message.content
            
            if content.startswith("Financing Agent:"):
                agent_name_list.append("Financing Agent")
            elif content.startswith("QA Agent:"):
                agent_name_list.append("QA Agent")
            elif content.startswith("Wrong Sample Analysis:"):
                agent_name_list.append("Wrong Sample Analysis")
            elif content.startswith("Annotator Profile Update:"):
                agent_name_list.append("Annotator Profile Update")
            elif content.startswith("Scheduling Agent:"):
                agent_name_list.append("Scheduling Agent")
            else:
                st.error(f"Unknown AI message content: {content}")

        elif isinstance(message, ToolMessage):
            if message.content.startswith("Annotation Agent:"):
                agent_name_list.append("Annotation Agent")
    if status == "Finished":
        agent_name_list.append("end")

    if len(agent_name_list) > 1 and agent_name_list[-1] == "Scheduling Agent" and status == "Running":
        agent_name_list.append("Annotation Agent")
    return agent_name_list


def get_total_round_cnt(backend_path, task_name):
    task_dir = os.path.join(backend_path, "output", task_name)
    round_dirs = os.listdir(task_dir)
    round_dirs = [d for d in round_dirs if d.startswith("round")]
    return len(round_dirs)

def get_last_chian(agents_messages):
    last_qa_message = None
    last_financing_message = None
    last_qa_chain = None
    last_financing_chain = None
    for message in agents_messages[1:]:
        if isinstance(message, AIMessage):
            if message.content.startswith("QA Agent:"):
                last_qa_message = message
            elif message.content.startswith("Financing Agent:"):
                last_financing_message = message
    
    if last_qa_message is not None:
        auto_lines, _ = crowdagent_tool.read_auto_and_llm_lines(last_qa_message.content)
        last_qa_chain = auto_lines[-1]
    if last_financing_message is not None:
        first_two_lines = last_financing_message.content.split("\n")[:2]
        last_financing_chain = first_two_lines[1]
    return last_qa_chain, last_financing_chain

def get_annotator_selection(backend_path, task_name, round_cnt):


    map_dict = {
        "llm_annotate_tool": "GPT-4o mini Annotator",
        "text_slm_annotate_tool": "RoBERTa Annotator",
        "human_annotate_tool": "Human Annotator",
        "multi_modal_slm_annotate_tool": "MMBT Annotator",
        "visual_slm_annotate_tool": "CovNext V2 Annotator",
        "vlm_annotate_tool": "GPT-4o mini(vlm) Annotator",
    }


    with open(f"{backend_path}/output/{task_name}/task_extra_info.yml", "r", encoding="utf8") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        str_messages = data["chunk"]
        dict_messages = eval(str_messages)
        messages = dict_messages["messages"]

        now_cnt = 1
        for message in messages:
            if isinstance(message, AIMessage):
                if message.content.startswith("Financing Agent:"):
                    now_cnt += 1
                elif message.content.startswith("Scheduling Agent:"):
                    if now_cnt == round_cnt:
                        try:
                            tool_name = message.tool_calls[0]["name"]
                            return map_dict[tool_name], tool_name
                        except Exception as e:
                            pass
                    elif now_cnt > round_cnt:
                        return None, None
        return None, None



def get_agents_messages_by_round(round_cnt):
    with open("asserts/info.txt", "r", encoding="utf8") as f:
        content = f.read()
        dict_content = eval(content)
    message_dict = {}
    round_num = 0
    for message in dict_content["messages"]:
        if isinstance(message, ToolMessage):
            round_num += 1
        if round_num not in message_dict:
            message_dict[round_num] = []
        message_dict[round_num].append(message)
    return message_dict[round_cnt]


def read_annotation_rules(task_output_dir, round_cnt):
    with open(f"{task_output_dir}/round{round_cnt}/annotation_rules.txt", "r", encoding="utf8") as f:
        content = f.read()
    return content


def get_average_confidence(line):
    return f"{float(line.split('Average confidence: ')[1]):.3f}"


def get_accuracy_and_unconverged_count(aline):
    if not aline:
        return "Waiting to be evaluated.", "Waiting to be evaluated."
    line = aline[:-3]
    parts = line.split(":")[1:]
    line = ":".join(parts)
    entries = line.split("=>")
    last_entry = entries[-1]
    last_entry_parts = last_entry.strip().strip("()").split(",")
    accuracy = float(last_entry_parts[1])
    unconverged_count = int(last_entry_parts[2])
    return accuracy, unconverged_count


def read_cost_and_left_budget(line):
    cost = float(line.split("This round cost")[1].split("$")[0])
    left_budget = float(line.split("remaining budget")[1].split("$")[0])
    cost = round(cost, 3)
    left_budget = round(left_budget, 3)
    return cost, left_budget


def calculate_budget_saved(aline, backend_path, task_name, round_cnt, now_acc=-1):
    try:
        line = aline[:-3]
        parts = line.split(":")
        line = ":".join(parts[1:])
        cost_entries = line.split("=>")
        total_cost = 0
        for round_num, entry in enumerate(cost_entries, 1):
            agent, cost, remaining_budget = entry.strip().strip("()").split(",")
            total_cost += float(cost)

        with open(f"{backend_path}/output/{task_name}/round1/extra_info.yml", "r", encoding="utf8") as f:
            content = f.read()
            dict_content = yaml.load(content, Loader=yaml.FullLoader)
            total_label_num = dict_content["label_count"]

        if now_acc == -1:
            with open(f"{backend_path}/output/{task_name}/round{round_cnt}/extra_info.yml", "r", encoding="utf8") as f:
                content = f.read()
                dict_content = yaml.load(content, Loader=yaml.FullLoader)
                now_acc = dict_content["accuracy"]
            
            
        human_price = 0.015
        human_budget = total_label_num * now_acc * human_price
        saved_budget = human_budget - total_cost
        return f"{saved_budget:.3f}", f"{total_cost:.3f}"
    except Exception as e:
        return "Waiting to be evaluated.", "Waiting to be evaluated."


def draw_budget_plot(aline):
    line = aline[:-3]
    parts = line.split(":")
    line = ":".join(parts[1:])
    cost_entries = line.split("=>")
    total_cost_list = []
    total_cost = 0
    for round_num, entry in enumerate(cost_entries, 1):
        agent, cost, remaining_budget = entry.strip().strip("()").split(",")
        total_cost += float(cost)
        total_cost_list.append(total_cost)

    chart_data = pd.DataFrame(
        {
            "round_num": range(1, len(total_cost_list) + 1),
            "total_cost": total_cost_list
        }
    )
    st.line_chart(chart_data, x="round_num", y="total_cost")


def draw_quality_plot(line):
    line = line[24:-3]
    entries = line.split("=>")
    total_accuracy_list = []
    total_unconverged_list = []
    for round_num, entry in enumerate(entries, 1):
        agent, accuracy, unconverged = entry.strip().strip("（）").replace("，", ",").split(",")
        total_accuracy_list.append(float(accuracy))
        total_unconverged_list.append(int(unconverged))

    chart_data = pd.DataFrame(
        {
            "round_num": range(1, len(total_accuracy_list) + 1),
            "total_accuracy": total_accuracy_list,
            "total_unconverged": total_unconverged_list
        }
    )
    st.line_chart(chart_data, x="round_num", y=["total_accuracy", "total_unconverged"])




def draw_unconverged_plot(line):
    line = line[24:-3]
    entries = line.split("=>")
    total_unconverged_list = []
    for round_num, entry in enumerate(entries, 1):
        agent, accuracy, unconverged = entry.strip().strip("（）").replace("，", ",").split(",")
        total_unconverged_list.append(int(unconverged))

    chart_data = pd.DataFrame(
        {
            "round_num": range(1, len(total_unconverged_list) + 1),
            "total_unconverged": total_unconverged_list
        }
    )
    st.line_chart(chart_data, x="round_num", y="total_unconverged")


def draw_accuracy_plot(line):
    line = line[24:-3]
    entries = line.split("=>")
    total_accuracy_list = []
    for round_num, entry in enumerate(entries, 1):
        agent, accuracy, unconverged = entry.strip().strip("（）").replace("，", ",").split(",")
        total_accuracy_list.append(float(accuracy))

    chart_data = pd.DataFrame(
        {
            "round_num": range(1, len(total_accuracy_list) + 1),
            "total_accuracy": total_accuracy_list
        }
    )
    st.line_chart(chart_data, x="round_num", y="total_accuracy")


def read_auto_and_llm_lines(raw_content):
    all_lines = raw_content.split("\n")
    auto_lines = []
    llm_lines = []
    auto_flag = True
    for line in all_lines:
        if auto_flag:
            auto_lines.append(line)
        else:
            line = line.replace("~", "\\~")
            llm_lines.append(line)
        if line.startswith("call path(agent, accuracy, unconverged count)"):
            auto_flag = False
    return auto_lines, llm_lines


def read_confidence_plot(task_output_dir, round_cnt):
    df = pd.read_csv(f"{task_output_dir}/round{round_cnt}/train_aggregated.csv")
    confidence = df.iloc[:,4]

    min_confidence = min(confidence)
    max_confidence = max(confidence)
    
    hist, bin_edges = np.histogram(confidence, bins=100, range=(0, 1))
    
    chart_data = pd.DataFrame({
        'confidence_interval': [f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' for i in range(len(bin_edges)-1)],
        'sample_count': hist
    })
    
    st.line_chart(chart_data, x='confidence_interval', y='sample_count')


def write_two_text(orange_text, black_text, padding="9px", font_size="1.2em"):
    st.markdown(f"""
<div style='padding-top: {padding}'>
    <span style='font-weight: normal; font-size: {font_size}; color: #BA750D;'>{orange_text}</span>
    <span style='font-weight: normal; font-size: {font_size};'>{black_text}</span>
</div>
""", unsafe_allow_html=True)
    

def write_text(text, padding="9px", font_size="1.2em", bold=True, color="black"):
    if bold:
        bold = "bold"
    else:
        bold = "normal"
    st.markdown(f"<div style='font-weight: {bold}; font-size: {font_size}; padding-top: {padding}; color: {color};'>{text}</div>", unsafe_allow_html=True)

    
def get_annotator(round_path):
    try:
        with open(os.path.join(round_path, "extra_info.yml"), "r", encoding="utf8") as f:
            content = f.read()
            dict_content = yaml.load(content, Loader=yaml.FullLoader)
            return dict_content["agent_name"]
    except:
        return "Unknown"
    

def get_annotation_cnt(round_path):
    try:
        with open(os.path.join(round_path, "extra_info.yml"), "r", encoding="utf8") as f:
            content = f.read()
            dict_content = yaml.load(content, Loader=yaml.FullLoader)
            return dict_content["label_count"]
    except:
        return "Unknown"


def get_annotation_accuracy(round_path):
    try:
        with open(os.path.join(round_path, "extra_info.yml"), "r", encoding="utf8") as f:
            content = f.read()
            dict_content = yaml.load(content, Loader=yaml.FullLoader)
            return f"{dict_content['accuracy']:.3f}"
    except:
        return "Waiting to be evaluated."
    

def get_budget_cost(round_path):
    try:
        with open(os.path.join(round_path, "extra_info.yml"), "r", encoding="utf8") as f:
            content = f.read()
            dict_content = yaml.load(content, Loader=yaml.FullLoader)
            return f"${dict_content['budget_cost']:.3f}"
    except:
        return "Waiting to be evaluated."


def read_confusion_matrix(task_dir, round_cnt):
    try:
        with open(os.path.join(task_dir, "confusion_matrix.yml"), "r", encoding="utf8") as f:
            content = f.read()
            dict_content = yaml.load(content, Loader=yaml.FullLoader)
        ret = {}
        for key, value in dict_content.items():
            key_num = int(key.split("_")[0])
            if key_num == round_cnt:
                ret[key] = value
        return ret
    except Exception as e:
        return {}
    
def read_all_confusion_matrix(task_dir):
    try:
        with open(os.path.join(task_dir, "confusion_matrix.yml"), "r", encoding="utf8") as f:
            content = f.read()
            dict_content = yaml.load(content, Loader=yaml.FullLoader)
        return dict_content
    except Exception as e:
        return {}


def get_confusion_matrix_figure(confusion_matrix, height=0):
    predicted_labels = []
    true_labels = []
    n = int(math.sqrt(len(confusion_matrix)-1))
    data = []
    for i in range(n):
        for j in range(n):
            key = f"w_{i}_t_{j}"
            value = confusion_matrix.get(key, 0)
            for _ in range(value):
                predicted_labels.append("pred"+str(i))
                true_labels.append("true"+str(j))
    df = pd.DataFrame({
        "pred": predicted_labels,
        "true": true_labels
    })
    pred_order = [f"pred{k}" for k in range(n)]
    true_order = [f"true{k}" for k in range(n)]
    if height == 0:
            fig = px.density_heatmap(
            data_frame=df,
            y="true",
            x="pred",
            category_orders={
                "pred": pred_order,
                "true": true_order
            },
            text_auto=True,
        )
    else:
        fig = px.density_heatmap(
            data_frame=df,
            y="true",
            x="pred",
            category_orders={
                "pred": pred_order,
                "true": true_order
            },
            height=height,
            text_auto=True,
        )
    return fig

def draw_confusion_matrix(confusion_matrix, component_key):
    n = int(math.sqrt(len(confusion_matrix)-1))
    data = []
    for i in range(n):
        for j in range(n):
            key = f"w_{i}_t_{j}"
            value = confusion_matrix.get(key, 0)
            data.append([i, n-1-j, value])
    
    x_labels = [f"Pred {i}" for i in range(n)]
    y_labels = [f"True {n-1-i}" for i in range(n)]
    
    option = {
        "tooltip": {"position": "top"},
        "grid": {"height": "50%", "top": "10%"},
        "xAxis": {"type": "category", "data": x_labels, "splitArea": {"show": True}},
        "yAxis": {"type": "category", "data": y_labels, "splitArea": {"show": True}},
        "visualMap": {
            "min": 0,
            "max": max([d[2] for d in data]),
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": "27%",
        },
        "series": [
            {
                "name": "Confusion Matrix",
                "type": "heatmap",
                "data": data,
                "label": {"show": True},
                "emphasis": {
                    "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.5)"}
                },
            }
        ],
    }
    
    st_echarts(option, height="500px", key=component_key)



def get_latest_train_aggregated(backend_path, task_name):
    task_dir = os.path.join(backend_path, "output", task_name)
    round_dirs = os.listdir(task_dir)
    round_dirs = [d for d in round_dirs if d.startswith("round")]
    round_dirs.sort(key=lambda x: int(x.split("round")[1]))

    final_df = None

    for round_dir in round_dirs:
        csv_file_path = os.path.join(task_dir, round_dir, "train_aggregated.csv")
        if os.path.exists(csv_file_path):
            with open(csv_file_path, 'r') as file:
                reader = csv.reader(file)
                header = next(reader)
                data = [row for row in reader]

            data_list = []
            for row in data:
                data_list.append({header[i]: row[i] for i in range(len(header))})

            df = pd.DataFrame(
                data_list
            )
            final_df = df

    return final_df


def get_text_label_df(backend_path, task_name, raw_df, strr=True):
    with open(os.path.join(backend_path, "config", "prompt.yml"), "r", encoding="utf8") as f:
        content = f.read()
        dict_content = yaml.load(content, Loader=yaml.FullLoader)
        category_to_label = dict_content[task_name]["category_to_label"]

    new_df = copy.deepcopy(raw_df)
    new_df = new_df.drop(columns=["index"])
    exception_col_name = ["index", "image_path", "text", "confidence"]
    for col_name in new_df.columns:
        if col_name not in exception_col_name:
            for key, value in category_to_label.items():
                if value in new_df[col_name]:
                    if strr:
                        new_df[col_name] = new_df[col_name].replace(str(value), key)
                    else:
                        new_df[col_name] = new_df[col_name].replace(value, key)
    return new_df


def convert_human_labeled_df(backend_path, task_name, raw_df):
    with open(os.path.join(backend_path, "config", "prompt.yml"), "r", encoding="utf8") as f:
        content = f.read()
        dict_content = yaml.load(content, Loader=yaml.FullLoader)
        category_to_label = dict_content[task_name]["category_to_label"]
    
    new_df = copy.deepcopy(raw_df)
    exception_col_name = ["index", "image_path", "text", "confidence"]

    for col_name in new_df.columns:
        if col_name not in exception_col_name:
            for key, value in category_to_label.items():
                if key in new_df[col_name].values:
                    new_df[col_name] = new_df[col_name].replace(key, str(value))

    return new_df

def get_display_df(backend_path, task_name, text_label_df):
    new_df = copy.deepcopy(text_label_df)
    token = "xxxxxxxxxxxxxxxxxxxxxxxxxx"  #  TODO： set your own token
    secret_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  #  TODO： set your own secret key
    new_df["image_path"] = new_df["image_path"].apply(lambda x: f"xxxxxxxxx/img?pth={encrypt_path(os.path.join(backend_path, 'data', task_name, 'images', x), secret_key)}&token={token}") # TODO: set your own url
    return new_df
    


def get_round_aggregated_data(round_cnt, task_output_dir):
    try:
        df = pd.read_csv(f"{task_output_dir}/round{round_cnt}/train_aggregated.csv")
        return df
    except Exception as e:
        return None


def get_annotation_path(data_str, cost_str):
    if not data_str or not cost_str:
        table_data = []
        return pd.DataFrame(table_data, columns=["Round", "Agent", "Accuracy", "Unconverged", "Cost", "Total Cost"])
    cost_line = cost_str[:-3]
    parts = cost_line.split(":")[1:]
    cost_line = ":".join(parts)
    cost_entries = cost_line.split("=>")

    data_line = data_str[:-3]
    parts = data_line.split(":")[1:]
    data_line = ":".join(parts)
    entries = data_line.split("=>")


    total_cost = 0
    table_data = []
    for round_num, entry in enumerate(entries, 1):
        agent, accuracy, unconverged = entry.strip().strip("()").split(",")
        try:
            agent, cost, _ = cost_entries[round_num - 1].strip().strip("()").split(",")
        except Exception as e:
            cost = 0
            
        total_cost += float(cost)

        table_data.append([
            round_num,
            agent,
            float(accuracy),
            int(unconverged),
            float(cost),
            float(total_cost)
        ])

    df = pd.DataFrame(table_data, columns=["Round", "Agent", "Accuracy", "Unconverged", "Cost", "Total Cost"])
    return df


def draw_acc_df(df):
    acc_df = pd.DataFrame(dict(
        Round = [f"{round_num}" for round_num, agent in enumerate(df["Agent"], 1)],
        Accuracy = df["Accuracy"]
    ))
    fig = px.line(acc_df, x="Round", y="Accuracy", text="Accuracy")
    fig.update_traces(textposition="bottom right")
    st.plotly_chart(fig)

def draw_unconv_df(df):
    unconv_df = pd.DataFrame(dict(
        Round = [f"{round_num}" for round_num, agent in enumerate(df["Agent"], 1)],
        Unconverged_Samples = df["Unconverged"]
    ))
    fig = px.line(unconv_df, x="Round", y="Unconverged_Samples", text="Unconverged_Samples")
    fig.update_traces(textposition="top right")
    st.plotly_chart(fig)

def draw_cost_df(df):
    df["Total Cost"] = df["Total Cost"].round(2)
    cost_df = pd.DataFrame(dict(
        Round = [f"{round_num}" for round_num, agent in enumerate(df["Agent"], 1)],
        Total_Cost = df["Total Cost"]
    ))
    fig = px.line(cost_df, x="Round", y="Total_Cost", text="Total_Cost")
    fig.update_traces(textposition="bottom right")
    st.plotly_chart(fig)


def append_dict_to_yml(yml_file_path, dict_data):
    if os.path.exists(yml_file_path):
        with open(yml_file_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            if data is None:
                data = {}
    else:
        data = {}

    data.update(dict_data)

    with open(yml_file_path, 'w') as file:
        yaml.dump(data, file)


def get_task_id(backend_path, task_name):
    task_name = task_name.replace(" ", "_")
    task_name = task_name.replace(".", "_")
    task_name = task_name.replace("-", "_")
    task_name = task_name.replace(":", "_")
    task_name = task_name.replace(";", "_")
    if os.path.exists(f"{backend_path}/output/{task_name}"):
        i = 1
        while os.path.exists(f"{backend_path}/output/{task_name}_{i}"):
            i += 1
        task_name = f"{task_name}_{i}"
    return task_name


def convert_labeled_gt_file(input_file, output_file, label_dict):
    with open(input_file, 'r', encoding='utf8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        data = [row for row in reader]

    tmp_data = copy.deepcopy(data)
    for i in range(len(tmp_data)):
        if tmp_data[i][1] in label_dict:
            tmp_data[i][1] = str(label_dict[tmp_data[i][1]])
        else:
            st.error(f"Label {tmp_data[i][1]} not found in label_dict, please check your label_dict.")

    with open(output_file, 'w', encoding='utf8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        for row in tmp_data:
            writer.writerow(row)

def submit_task(session_state):

    api_key_config = {
        "open_ai":{
            "api_key": session_state.openai_api_key_value,
            "base_url": session_state.openai_base_url_value,
            "model_name": "gpt-4o-mini",
        }
    }

    llm_config = {
        f"{session_state.task_id_value}": {
            "PROJECT_ROOT": session_state.backend_path,
            "NUM_LABELS": len(session_state.custom_label_value),
            "BUDGET": session_state.budget_value,
            "DETAIL_LABELS": {label_name: label_id for label_id, label_name in enumerate(session_state.custom_label_value)}
        }
    }

    prompt_config = {
        f"{session_state.task_id_value}": {
            "human_request": session_state.task_description_value,
            "llm_annotation": session_state.prompt_template_data1,
            "category_to_label": {label_name: label_id for label_id, label_name in enumerate(session_state.custom_label_value)},
            "annotation_agent": session_state.agent_options_value,
        }
    }

    slm_config = {
        f"{session_state.task_id_value}": {
            "GPUID": 0,
            "SEED": 42,
            "TRAIN_BATCH": 32,
            "N_GPU": 1,
            "NUM_LABELS": len(session_state.custom_label_value),
            "RHO_SEL": 0.2,
            "SELECT_DEMO_NUM": 10,
            "distbert_epoches": 20,
            "roberta_epoches": 20,
            "roberta_batch": 16,
            "convnextv2_epoches": 50,
            "mmbt_epoches": 10,
            "TAU": 0.5,
            "warmup_port": 3,
            "num_image_embeds": 3,
            "max_seq_length": 512,
        }
    }
    append_dict_to_yml(f"{session_state.backend_path}/config/api_key.yml", api_key_config)
    append_dict_to_yml(f"{session_state.backend_path}/config/llm_config.yml", llm_config)
    append_dict_to_yml(f"{session_state.backend_path}/config/prompt.yml", prompt_config)
    append_dict_to_yml(f"{session_state.backend_path}/config/slm_config.yml", slm_config)

    with st.spinner("Creating environment just for you, please wait...", show_time=True):
        if session_state.use_example_files:
            convert_labeled_gt_file(
                "asserts/example_files/labeled_gt.csv", 
                "asserts/example_files/labeled_gt_tmp.csv",
                {label_name: label_id for label_id, label_name in enumerate(session_state.custom_label_value)}
            )
            bash_code = f"""mkdir -p {session_state.backend_path}/data/{session_state.task_id_value} 
echo 'dir created' 
cp -f asserts/example_files/labeled_gt_tmp.csv {session_state.backend_path}/data/{session_state.task_id_value}/labeled_gt.csv 
echo 'labeled_gt.csv copied' 
cp -f asserts/example_files/train.csv {session_state.backend_path}/data/{session_state.task_id_value}/train.csv 
echo 'train.csv copied' 
cp -f asserts/example_files/images.zip {session_state.backend_path}/data/{session_state.task_id_value}/images.zip 
echo 'images.zip copied' 
unzip -o {session_state.backend_path}/data/{session_state.task_id_value}/images.zip -d {session_state.backend_path}/data/{session_state.task_id_value}
echo 'images unzipped' 
mkdir -p {session_state.backend_path}/output/{session_state.task_id_value} 
echo 'output dir created' 
cd {session_state.backend_path}
nohup python labelchain.py --task {session_state.task_id_value} > {session_state.backend_path}/output/{session_state.task_id_value}/nohup.log 2>&1 & 
echo $! > process.pid"""
            with open(f"run.sh", "w") as f:
                f.write(bash_code)
            os.system(f"bash run.sh")
        else:
            bash_code = f"""mkdir -p {session_state.backend_path}/data/{session_state.task_id_value} 
echo 'dir created' 
cp -f asserts/user_files/train.csv {session_state.backend_path}/data/{session_state.task_id_value}/train.csv 
echo 'train.csv copied' """
            if session_state.use_golden_samples:
                convert_labeled_gt_file(
                    "asserts/user_files/labeled_gt.csv", 
                    "asserts/user_files/labeled_gt_tmp.csv",
                    {label_name: label_id for label_id, label_name in enumerate(session_state.custom_label_value)}
                )
                bash_code += f"""cp -f asserts/user_files/labeled_gt_tmp.csv {session_state.backend_path}/data/{session_state.task_id_value}/labeled_gt.csv 
echo 'labeled_gt.csv copied' """
            if session_state.use_multi_modal:
                bash_code += f"""cp -f asserts/user_files/images.zip {session_state.backend_path}/data/{session_state.task_id_value}/images.zip 
echo 'images.zip copied'
unzip -o {session_state.backend_path}/data/{session_state.task_id_value}/images.zip -d {session_state.backend_path}/data/{session_state.task_id_value}
echo 'images unzipped' """
            bash_code += f"""mkdir -p {session_state.backend_path}/output/{session_state.task_id_value} 
echo 'output dir created' 
cd {session_state.backend_path}
nohup python labelchain.py --task {session_state.task_id_value} > {session_state.backend_path}/output/{session_state.task_id_value}/nohup.log 2>&1 & 
echo $! > process.pid"""
            with open(f"run.sh", "w") as f:
                f.write(bash_code)
            os.system(f"bash run.sh")



    st.success(f"Your annotation task has been created successfully!")
    st.html("""
    <br>
    """)
    with st.container(border=True):
        st.write(f"Your task id is:")
        crowdagent_tool.write_text(f"{session_state.task_id_value}", font_size="2em", bold=True, color="black")

        st.html("""
        <br>
        """)

    st.html("""
    <br>
    """)


def get_task_list(session_state):
    task_list = []
    output_dir = f"{session_state.backend_path}/output"
    for task_name in os.listdir(output_dir):
        task_list.append(task_name)
    task_list.sort()
    return task_list
        
def get_progress(task_dir, round_dir):
    try:
        with open(os.path.join(task_dir, round_dir, "extra_info.yml"), "r", encoding="utf8") as f:
                content = f.read()
                dict_content = yaml.load(content, Loader=yaml.FullLoader)
                return dict_content["progress"]
    except:
        return 0
    

def get_annotation_rules(backend_path, task_name, round_cnt):
    try:
        with open(f"{backend_path}/output/{task_name}/round{round_cnt}/annotation_rules.txt", "r", encoding="utf8") as f:
            content = f.read()
        return content
    except Exception as e:
        return "Annotation rules not available yet."
    

def check_annotation_running(backend_path, task_name, round_cnt):
    map_dict = {
        "gpt-4o-mini": "llm_annotate_tool",
        "RoBERTa": "text_slm_annotate_tool",
        "human": "human_annotate_tool",
        "MMBT": "multi_modal_slm_annotate_tool",
        "ConvNext v2": "visual_slm_annotate_tool",
        "gpt-4o-mini(visual)": "vlm_annotate_tool",
    }
    if not os.path.exists(os.path.join(backend_path, "output", task_name, f"round{round_cnt}")):
        return 1, "Unknown"
    try:
        extra_info_path = os.path.join(backend_path, "output", task_name, f"round{round_cnt}", "extra_info.yml")
        with open(extra_info_path, "r", encoding="utf8") as f:
            content = f.read()
            dict_content = yaml.load(content, Loader=yaml.FullLoader)
            progress = dict_content.get("progress", 0)
            agent_name = dict_content.get("agent_name", "Unknown")
            if agent_name in map_dict:
                agent_name = map_dict[agent_name]
            return progress, agent_name
    except Exception as e:
        return 0, "Unknown"
    

def get_annotation_results(backend_path, task_name, round_cnt):
    try:
        task_dir = os.path.join(backend_path, "output", task_name)
        files = os.listdir(os.path.join(task_dir, f"round{round_cnt}"))
        files = [file for file in files if file.endswith("csv") and file != "train_aggregated.csv" and file != "most_representative_idx.csv"]
        selected_file = files[0]
        df = pd.read_csv(os.path.join(task_dir, f"round{round_cnt}", selected_file))
        return df
    except Exception as e:
        return None
    

def check_task_configuration(session_state):
    if not session_state.openai_api_key_value:
        return False, "OpenAI API Key is required."
    return True, "Configuration is complete."