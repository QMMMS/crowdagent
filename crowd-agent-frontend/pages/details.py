import streamlit as st
import os
import pandas as pd
import numpy as np
import time
import util.crowdagent_tool as crowdagent_tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from streamlit_echarts import st_echarts
from st_circular_progress import CircularProgress
import io

st.set_page_config(layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.switch_page('pages/login.py')

st.markdown(""" <style>iframe[title="streamlit_echarts.st_echarts"]{ height: 400px !important } """, unsafe_allow_html=True)

with st.sidebar:
    st.image("files/logo.png")
    st.markdown("---")
    st.write("Page Navigation")
    pages = {
        "Configuration": {"icon": "🔧", "page": "home.py"},
        "Agents": {"icon": "🤖", "page": "pages/agent.py"},
        "Details": {"icon": "🔍", "page": "pages/details.py"},
        "Dashboard": {"icon": "📊", "page": "pages/dashboard.py"},
    }
    st.session_state.page = ""
    for page_name, page_info in pages.items():
        if st.sidebar.button(
            f"{page_info['icon']} {page_name}", key=page_name, use_container_width=True
        ):
            st.switch_page(page_info['page'])
    
    st.markdown("---")
    st.write("Account")
    if st.sidebar.button("⏻ Logout", key="logout", use_container_width=True):
        st.session_state.logged_in = False
        st.switch_page('pages/login.py')
    st.markdown("---")
    if st.sidebar.button("Refresh", key="refresh", use_container_width=True):
        st.rerun()

st.html(
    """
    <style>
    [data-testid="stVerticalBlock"] {
        gap: 0.4rem;
    }
    </style>
    """
)
st.session_state.backend_path = "xxxxxxxxxxxxxxxxx" # TODO: change to the backend path

@st.cache_data
def get_data(file_name):
    try:
        df = pd.read_csv(file_name)
    except Exception as e:
        st.error(f"Error reading file {file_name}: {e}")
        return pd.DataFrame()
    return df

@st.cache_data
def convert_for_download(df: pd.DataFrame):
    if not df.empty:
        text_label_df = crowdagent_tool.get_text_label_df(st.session_state.backend_path, task_name, df, strr=False)
        df = text_label_df.copy()
    return df.to_csv(index=False).encode("utf-8")

title_col1, title_col2 = st.columns(2)
with title_col1:
    st.title("Annotation Details")
    st.write("Select a round to view the details.")
with title_col2:
    st.html("""
    <br>
    """)
    task_name = st.selectbox(
        "Select an annotation task id to view the details",
        crowdagent_tool.get_task_list(st.session_state),
    )

st.markdown("---")

task_dir = os.path.join(st.session_state.backend_path, "output", task_name)
round_dirs = os.listdir(task_dir)
round_dirs = [d for d in round_dirs if d.startswith("round")]
round_dirs.sort(key=lambda x: int(x.split("round")[1]))

if len(round_dirs) == 0:
    with st.spinner("Task is initializing, please wait...", show_time=True):
        time.sleep(5)
    st.rerun()

round_tabs = st.tabs(round_dirs)

for round_tab, round_dir in zip(round_tabs, round_dirs):
    with round_tab:
        files = os.listdir(os.path.join(task_dir, round_dir))
        files = [file for file in files if file.endswith("csv") and file != "train_aggregated.csv" and file != "most_representative_idx.csv"]

        left, right = st.columns([1, 2])
        with right.container(border=True):
            st.subheader("Annotation Result")
            file_chosen = st.selectbox("Select a file", files, key=f"file_{round_dir}")
            if round_dir != round_dirs[-1]:
                raw_df = pd.read_csv(os.path.join(task_dir, round_dir, file_chosen))
                text_label_df = crowdagent_tool.get_text_label_df(st.session_state.backend_path, task_name, raw_df, strr=False)
                st.dataframe(text_label_df, hide_index=True, height=1000)
            else:
                place_holder = st.empty()
        
        with left:
            with st.container(border=True):
                st.subheader("Round Information")
                agent_name = crowdagent_tool.get_annotator(os.path.join(task_dir, round_dir))
                crowdagent_tool.write_two_text("Agent Name: ", agent_name, padding="15px", font_size="1.2em")
                crowdagent_tool.write_two_text("Label Count: ", crowdagent_tool.get_annotation_cnt(os.path.join(task_dir, round_dir)), padding="15px", font_size="1.2em")
                crowdagent_tool.write_two_text("Accuracy: ", crowdagent_tool.get_annotation_accuracy(os.path.join(task_dir, round_dir)), padding="15px", font_size="1.2em")
                crowdagent_tool.write_two_text("Budget Cost: ", crowdagent_tool.get_budget_cost(os.path.join(task_dir, round_dir)), padding="15px", font_size="1.2em")

            with st.container(border=True):
                if agent_name != "human":
                    if agent_name in ["RoBERTa", "MMBT", "ConvNext V2"]:
                        st.subheader("Training Progress")
                    else:
                        st.subheader("Annotation Progress")
                    if round_dir != round_dirs[-1]:
                        percentage = crowdagent_tool.get_progress(task_dir, round_dir) * 100
                        my_circular_progress = CircularProgress(
                            label="",
                            value=int(percentage),
                            key=f"liquidfill_progress_{round_dir}",
                            size="large",
                        )
                        my_circular_progress.st_circular_progress()
                        my_circular_progress.update_value(int(percentage))
                    else:
                        liquidfill_placeholder = st.empty()
                else:
                    st.subheader("Human Annotation")
                    df = get_data(os.path.join(task_dir, round_dir, "to_be_annotated.csv"))
                    csv = convert_for_download(df)
                    st.markdown("<br>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download unlabeled file",
                            data=csv,
                            file_name="to_be_annotated.csv",
                            mime="text/csv",
                            icon=":material/download:",
                            use_container_width=True,
                            key=f"download_unlabeled_file_{round_dir}"
                        )
                    with col2:
                        st.link_button("Go to Youling Platform", "https://zb.163.com/mark/task", icon=":material/link:", use_container_width=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    labeled_file = st.file_uploader("Upload labeled file", type="csv", key=f"upload_labeled_file_{round_dir}")
                    if labeled_file is not None:
                        dataframe = pd.read_csv(labeled_file)
                        temp_dataframe = crowdagent_tool.convert_human_labeled_df(
                            st.session_state.backend_path, task_name, dataframe
                        )
                        labeled_file_path = os.path.join(task_dir, round_dir, "human.csv")
                        temp_dataframe.to_csv(labeled_file_path, index=False)
                        st.success(f"Labeled file uploaded successfully")
                        
                    liquidfill_placeholder = None

            with st.container(border=True):
                confusion_matrix = crowdagent_tool.read_confusion_matrix(task_dir, int(round_dir.split("round")[1]))
                st.subheader("Confusion Matrix")
                if len(confusion_matrix) > 0:
                    confusion_matrix_key_list = list(confusion_matrix.keys())
                    key_chosen = st.selectbox("Select a confusion_matrix", confusion_matrix_key_list, key=f"confusion_matrix_key_{round_dir}")
                    crowdagent_tool.draw_confusion_matrix(confusion_matrix[key_chosen], f"confusion_matrix_{round_dir}")
                else:
                    st.write("Waiting to be evaluated.")

@st.fragment(run_every="2s")
def update_progress_bar1(place_holder, liquidfill_placeholder):
    with place_holder.container():
        if file_chosen:
            raw_df = pd.read_csv(os.path.join(task_dir, round_dir, file_chosen))
            text_label_df = crowdagent_tool.get_text_label_df(st.session_state.backend_path, task_name, raw_df, strr=False)
            display_df = crowdagent_tool.get_display_df(st.session_state.backend_path, task_name, text_label_df)
            st.dataframe(display_df, hide_index=True, height=1000)
        else:
            st.write("No files available.")
    if liquidfill_placeholder is None:
        return
    with liquidfill_placeholder.container():
        percentage = crowdagent_tool.get_progress(task_dir, round_dir) * 100
        my_circular_progress = CircularProgress(
            label="",
            value=int(percentage),
            key=f"liquidfill_progress_last",
            size="large",
        )
        my_circular_progress.st_circular_progress()
        my_circular_progress.update_value(int(percentage))
update_progress_bar1(place_holder, liquidfill_placeholder)
