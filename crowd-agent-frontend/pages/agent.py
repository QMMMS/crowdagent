import streamlit as st
from streamlit_echarts import st_echarts
import util.crowdagent_tool as crowdagent_tool
import time
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import os
import plotly.express as px
from component.agent_map_component import agent_map_component

st.set_page_config(layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.switch_page('pages/login.py')

round_cnt = 1
current_round_agents = []
current_round_messages = []
st.session_state.agent_name_list = []

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
    st.markdown(f"<div style='padding-top: 50px'></div>", unsafe_allow_html=True)
    is_expand = st.toggle("Expand")
    height_scale = 2 if is_expand else 1

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

title_col1, title_col2 = st.columns(2)
with title_col1:
    st.title("Agents Interaction")
    st.write("See the agents' messages and interactions here.")
with title_col2:
    st.html("""
    <br>
    """)
    task_name = st.selectbox(
        "Select an annotation task id to view the agents' messages:",
        crowdagent_tool.get_task_list(st.session_state),
    )

st.markdown("---")

agents_messages = crowdagent_tool.read_agents_messages(st.session_state.backend_path, task_name)

placeholder = st.empty() 

human_message = agents_messages[0]
if isinstance(human_message, HumanMessage):
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(human_message.content)

current_round_messages = []

def update_progress_bar(text):
    col1, col2 = st.columns([10, 80])
    with col1:
        st.write(text)
    with col2:
        progress_bar = st.progress(0)
        now_progress = 0
        while now_progress < 0.9:
            now_progress += 0.1
            progress_bar.progress(now_progress)
            time.sleep(0.2)
        progress_bar.progress(100)

def update_agent_tabs(current_round_messages, round_cnt):
    container = st.container(border=True)
    col1, col2 = container.columns([1,15], vertical_alignment="center")

    col1.markdown(f"## R{round_cnt}")
    
    agent_groups = {
        "🤖Scheduling Agent": [],
        "🔧Annotation Agent": [],
        "📊Wrong Sample Analysis": [],
        "🔍QA Agent": [],
        "👤Annotator Profile Update": [],
        "💰Financing Agent": [],
    }
    
    for message in current_round_messages:
        if isinstance(message, AIMessage):
            if message.content.startswith("Financing Agent:"):
                agent_groups["💰Financing Agent"].append(message)
            elif message.content.startswith("QA Agent:"):
                agent_groups["🔍QA Agent"].append(message)
            elif message.content.startswith("Wrong Sample Analysis:"):
                agent_groups["📊Wrong Sample Analysis"].append(message)
            elif message.content.startswith("Scheduling Agent:"):
                if len(agent_groups["🤖Scheduling Agent"]) == 0:
                    agent_groups["🤖Scheduling Agent"].append(message)
            elif message.content.startswith("Annotator Profile Update:"):
                agent_groups["👤Annotator Profile Update"].append(message)
            else:
                st.error(f"AAA Unknown AI message content: {message.content}")
        elif isinstance(message, ToolMessage):
            agent_groups["🔧Annotation Agent"].append(message)
        else:
            st.error(f"Unknown message type: {type(message)}")
    
    active_agents = [agent for agent, messages in agent_groups.items() if messages]
    
    if active_agents:
        tab_list = col2.tabs(active_agents)
        for i, tab in enumerate(tab_list):
            with tab:
                agent_name = active_agents[i]
                messages = agent_groups[agent_name]
                for message in messages:
                    display_agent_message(message, round_cnt)
    
def display_agent_message(message, round_cnt):
    if isinstance(message, AIMessage):
        if message.content.startswith("Financing Agent:"):
            with st.chat_message("ai", avatar="💰"):
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(height=320*height_scale, border=False):
                        raw_content = message.content
                        first_two_lines = raw_content.split("\n")[:2]
                        other_lines = raw_content.split("\n")[2:]
                        st.write('\n'.join(other_lines).replace("$", "\$"))
                with col2:
                    with st.container(height=320*height_scale, border=True):
                        cost, left_budget = crowdagent_tool.read_cost_and_left_budget(first_two_lines[0])
                        budget_saved, _ = crowdagent_tool.calculate_budget_saved(first_two_lines[1], st.session_state.backend_path, task_name, round_cnt)
                        crowdagent_tool.write_two_text("Budget Cost: ", f"${cost}", padding="15px", font_size="1.2em")
                        crowdagent_tool.write_two_text("Remaining Budget: ", f"${left_budget}", padding="15px", font_size="1.2em")
                        crowdagent_tool.write_two_text("Budget Saved: ", f"${budget_saved}", padding="15px", font_size="1.2em")
                        st.markdown("---")
                        st.subheader("Budget Plot")
                        crowdagent_tool.draw_budget_plot(first_two_lines[1])

        elif message.content.startswith("QA Agent:"):
            raw_content = message.content
            auto_lines, llm_lines = crowdagent_tool.read_auto_and_llm_lines(raw_content)
            with st.chat_message("ai", avatar="🔍"):
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(height=300*height_scale, border=False):
                        st.write("\n".join(llm_lines))
                with col2:
                    with st.container(height=300*height_scale, border=True):
                        average_confidence = crowdagent_tool.get_average_confidence(auto_lines[4])
                        acc, unconverged_count = crowdagent_tool.get_accuracy_and_unconverged_count(auto_lines[-1])
                        crowdagent_tool.write_two_text("Accuracy: ", f"{acc}", padding="15px", font_size="1.2em")
                        crowdagent_tool.write_two_text("Unconverged Count: ", f"{unconverged_count}", padding="15px", font_size="1.2em")
                        crowdagent_tool.write_two_text("Average Confidence: ", f"{average_confidence}", padding="15px", font_size="1.2em")
                        
                        st.markdown("---")
                        st.subheader("Confidence Distribution")
                        df = crowdagent_tool.get_round_aggregated_data(round_cnt, os.path.join(st.session_state.backend_path, "output", task_name))
                        fig = px.histogram(data_frame=df, x="confidence")
                        st.plotly_chart(fig, key=f"confidence_histogram_{round_cnt}")
                        
        elif message.content.startswith("Wrong Sample Analysis:"):
            with st.chat_message("assistant", avatar="📊"):
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(height=300*height_scale, border=False):
                        message_content_list = message.content.split(":")
                        st.write(":".join(message_content_list[1:]))
                with col2:
                    with st.container(height=300*height_scale, border=True):
                        rules = crowdagent_tool.get_annotation_rules(st.session_state.backend_path, task_name, round_cnt)
                        st.markdown(rules)

        elif message.content.startswith("Annotator Profile Update:"):
            with st.chat_message("ai", avatar="👤"):
                col1, col2 = st.columns([1,1])
                with col1:
                    with st.container(height=320*height_scale, border=False):
                        message_content_list = message.content.split(":")
                        st.write(":".join(message_content_list[1:]))
                with col2:
                    with st.container(height=320*height_scale, border=True):
                        st.subheader("Confusion Matrix")
                        task_dir = os.path.join(st.session_state.backend_path, "output", task_name)
                        confusion_matrix = crowdagent_tool.read_confusion_matrix(task_dir, round_cnt)
                        key_chosen = list(confusion_matrix.keys())[0]
                        fig = crowdagent_tool.get_confusion_matrix_figure(confusion_matrix[key_chosen], height=300 if height_scale == 1 else 500)
                        st.plotly_chart(fig, use_container_width=True, key=f"confusion_matrix_plot_{round_cnt}")

        elif message.content.startswith("Scheduling Agent:"):
            with st.chat_message("ai", avatar="🤖"):
                if height_scale == 1:
                    col1, col2 = st.columns([1,1])
                    with col1:
                        with st.container(height=300, border=False):
                            message_content_list = message.content.split(":")
                            st.write(":".join(message_content_list[1:]))
                    with col2:
                        with st.container(height=300, border=True):
                            annotator_selection, tool_name = crowdagent_tool.get_annotator_selection(st.session_state.backend_path, task_name, round_cnt)
                            st.write(f"Annotator Selection: {annotator_selection}")
                            st.markdown(f"<div style='padding-top: 50px'></div>", unsafe_allow_html=True)
                            if os.path.exists(f"files/{tool_name}.jpg"):
                                st.image(f"files/{tool_name}.jpg", width=400)
                else:
                    with st.container(height=500, border=False):
                        message_content_list = message.content.split(":")
                        st.write(":".join(message_content_list[1:]))

        else:
            st.error(f"BBB Unknown AI message content: {message.content}")
    elif isinstance(message, ToolMessage):
        with st.chat_message("tool", avatar="🔧"):
            if height_scale == 1:
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(height=320, border=False):
                        if message.content.startswith("Progress:"):
                            with st.container(border=True):
                                st.write("Annotator Working... Please wait.")
                                progress = float(message.content.split(":")[1].strip())
                                st.progress(progress)
                        else:
                            with st.status(f"Annotator working...", expanded=True) as status:
                                    status.update(
                                        label=message.content, state="complete", expanded=False
                                    )
                        st.markdown(f"<div style='padding-top: 80px'></div>", unsafe_allow_html=True)
                        tool_name = message.name
                        if os.path.exists(f"files/{tool_name}.jpg"):
                            st.image(f"files/{tool_name}.jpg", width=400)

                with col2:
                    with st.container(height=320, border=True):
                        annotation_results = crowdagent_tool.get_annotation_results(st.session_state.backend_path, task_name, round_cnt)
                        if annotation_results is not None:
                            st.subheader("Annotation Results")
                            text_label_df = crowdagent_tool.get_text_label_df(st.session_state.backend_path, task_name, annotation_results, strr=False)
                            display_df = crowdagent_tool.get_display_df(st.session_state.backend_path, task_name, text_label_df)
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
                                height=300,
                            )
                        else:
                            st.write("No annotation results available yet.")
            else:
                if message.content.startswith("Progress:"):
                    with st.container(border=True):
                        st.write("Annotator Working... Please wait.")
                        progress = float(message.content.split(":")[1].strip())
                        st.progress(progress)
                else:
                    with st.status(f"Annotator working...", expanded=True) as status:
                            status.update(
                                label=message.content, state="complete", expanded=False
                            )
                annotation_results = crowdagent_tool.get_annotation_results(st.session_state.backend_path, task_name, round_cnt)
                if annotation_results is not None:
                    tabel_scale = 1
                else:
                    tabel_scale = 0.1
                with st.container(height=int(300*height_scale*tabel_scale), border=True):
                    if annotation_results is not None:
                            st.subheader("Annotation Results")
                            text_label_df = crowdagent_tool.get_text_label_df(st.session_state.backend_path, task_name, annotation_results, strr=False)
                            display_df = crowdagent_tool.get_display_df(st.session_state.backend_path, task_name, text_label_df)
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
                                height=300*height_scale,
                            )
                    else:
                        st.write("No annotation results available yet.")
                        
    else:
        st.error(f"Unknown message type: {type(message)}")

for message in agents_messages[1:]:
    if isinstance(message, AIMessage):
        content = message.content
        
        current_round_messages.append(message)
        
        if content.startswith("Financing Agent:"):
            st.session_state.agent_name_list.append("Financing Agent")
            update_agent_tabs(current_round_messages, round_cnt)
            round_cnt += 1
            current_round_messages = []
        elif content.startswith("QA Agent:"):
            st.session_state.agent_name_list.append("QA Agent")
        elif content.startswith("Wrong Sample Analysis:"):
            st.session_state.agent_name_list.append("Wrong Sample Analysis")
        elif content.startswith("Annotator Profile Update:"):
            st.session_state.agent_name_list.append("Annotator Profile Update")
        elif content.startswith("Scheduling Agent:"):
            st.session_state.agent_name_list.append("Scheduling Agent")

        else:
            st.error(f"CCC Unknown AI message content: {content}")

    elif isinstance(message, ToolMessage):
        if message.content.startswith("Annotation Agent:"):
            current_round_messages.append(message)
            st.session_state.agent_name_list.append("Annotation Agent")

last_progress, tool_name = crowdagent_tool.check_annotation_running(st.session_state.backend_path, task_name, round_cnt)
if last_progress != 1:
    st.session_state.agent_name_list.append("Annotation Agent")
    current_round_messages.append(
        ToolMessage(
            f"Progress: {last_progress:.2f}",
            tool_call_id="tool_call_id",
            name=tool_name
        )
    )

if current_round_messages:
    update_agent_tabs(current_round_messages, round_cnt)

status = crowdagent_tool.get_task_status(st.session_state.backend_path, task_name)
if status == "Finished":
    st.session_state.agent_name_list.append("end")

with placeholder.container():
    agent_map_component(
        agent_name=st.session_state.agent_name_list,
        key=f"agent_map_component_{round_cnt}"
    )
