import streamlit as st
import time
import util.crowdagent_tool as crowdagent_tool
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from component.agent_map_component import agent_map_component
import os

st.set_page_config(layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.switch_page('pages/login.py')

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

st.session_state.backend_path = "xxxxxxxxxxxxxxxxx"  # TODO: change to the backend path

title_col1, title_col2 = st.columns(2)

with title_col1:
    st.title("Dashboard")
    st.write("Monitor your annotation task here.")
with title_col2:
    st.html("""
    <br>
    """)
    task_name = st.selectbox(
        "Select an annotation task id to monitor:",
        crowdagent_tool.get_task_list(st.session_state),
    )

st.markdown("---")
agents_messages = crowdagent_tool.read_agents_messages(st.session_state.backend_path, task_name)
total_round_cnt = crowdagent_tool.get_total_round_cnt(st.session_state.backend_path, task_name)
last_qa_chian, last_financing_chian = crowdagent_tool.get_last_chian(agents_messages)
chain_df = crowdagent_tool.get_annotation_path(last_qa_chian, last_financing_chian)
status = crowdagent_tool.get_task_status(st.session_state.backend_path, task_name)

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns([2,1,1,1,1,1])

kpi1.metric(
    label="Task Status 📝",
    value=status,
)

kpi2.metric(
    label="Round Count ⏳",
    value=total_round_cnt,
)

accuracy, unconverged_count = crowdagent_tool.get_accuracy_and_unconverged_count(last_qa_chian)

kpi3.metric(
    label="Accuracy 🎯",
    value=accuracy,
)

kpi4.metric(
    label="Unconverged 🔄",
    value=unconverged_count,
)

saved_budget, total_cost = crowdagent_tool.calculate_budget_saved(last_financing_chian, st.session_state.backend_path, task_name, total_round_cnt, accuracy)

kpi5.metric(
    label="Total Cost 💵",
    value='$'+str(total_cost),
)

kpi6.metric(
    label="Budget Saved 💰",
    value='$'+str(saved_budget),
)

st.html("""
<br>
""")

with st.container(border=True):
    st.subheader("Agent Map")
    agent_name_list = crowdagent_tool.get_agent_name_list(agents_messages, status=status)
    placeholder = st.empty()
    with placeholder.container():
        agent_map_component(
            agent_name=agent_name_list,
        )

if total_round_cnt > 1:

    with st.container(border=True):
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.subheader("Metrics")
            options = ["Accuracy", "Unconverged Samples", "Total Cost"]
            selection = st.segmented_control(
                "Metrics", options, selection_mode="single", default ="Accuracy", key="info_tracing_selection", label_visibility="hidden"
            )
            if selection == "Accuracy":
                crowdagent_tool.draw_acc_df(chain_df)
            elif selection == "Unconverged Samples":
                crowdagent_tool.draw_unconv_df(chain_df)
            elif selection == "Total Cost":
                crowdagent_tool.draw_cost_df(chain_df)
        with info_col2:
            st.subheader("Confidence Distribution")
            round_cnt = st.slider("Select a round to visualize the confidence distribution.", 1, total_round_cnt, 1)
            df = crowdagent_tool.get_round_aggregated_data(round_cnt, os.path.join(st.session_state.backend_path, "output", task_name))
            if df is None:
                st.write("Waiting for the round to finish...")
            else:
                fig = px.histogram(data_frame=df, x="confidence")
                st.plotly_chart(fig)

    with st.container(border=True):
        left_col, right_col = st.columns(2)
        with left_col:
            st.subheader("Confusion Matrix")
            confusion_matrix = crowdagent_tool.read_all_confusion_matrix(os.path.join(st.session_state.backend_path, "output", task_name))
            confusion_keys = list(confusion_matrix.keys())
            confusion_keys.sort(key=lambda x: int(x.split("_")[0]))
            confusion_key_chosen = st.selectbox(
                "Select a round to visualize the confusion matrix:",
                confusion_keys,
                label_visibility="hidden"
            )
            fig = crowdagent_tool.get_confusion_matrix_figure(confusion_matrix[confusion_key_chosen])
            st.plotly_chart(fig)

        with right_col:
            st.subheader("Annotation path")
            st.dataframe(chain_df, hide_index=True)

    with st.container(border=True):
        raw_df = crowdagent_tool.get_latest_train_aggregated(st.session_state.backend_path, task_name)
        text_label_df = crowdagent_tool.get_text_label_df(st.session_state.backend_path, task_name, raw_df)
        display_df = crowdagent_tool.get_display_df(st.session_state.backend_path, task_name, text_label_df)
        col1, col2, col3 = st.columns(3)
        col1.subheader("Annotated Data")
        col3.download_button(
            label="Download annotated data",
            data=raw_df.to_csv(index=False),
            file_name="annotated_data.csv",
            mime="text/csv",
            icon=":material/download:",
            use_container_width=True,
            key=f"download_annotated_data"
        )
        
        st.data_editor(
            display_df,
            hide_index=True,
            column_config={
                "image_path": st.column_config.LinkColumn(
                    "Image Preview",
                    display_text="Image Preview",
                    help="The image preview",
                ),
            },
        )

else:
    with st.container(border=True):
        st.write("## More metrics are waiting to be evaluated....")
