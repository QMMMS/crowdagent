import streamlit as st
import util.read_info as read_info
import pandas as pd
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


st.html(
    """
    <style>
    [data-testid="stVerticalBlock"] {
        gap: 0.4rem;
    }
    </style>
    """
)

st.title("Task Configuration")
st.write("Update your annotation task configuration here.")

st.markdown("---")

if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.task_submitted = False
    st.session_state.backend_path = "" # TODO: change to the backend path
    st.session_state.task_name_value = '' # TODO: default task name
    st.session_state.budget_value = 0 # TODO: default budget
    st.session_state.task_description_value = "" # TODO: default task description
    st.session_state.openai_api_key_value = "" # TODO: openai api key
    st.session_state.openai_base_url_value = "" # TODO: openai base url
    st.session_state.custom_label_value = [] # TODO: default custom label
    st.session_state.agent_options_value = [] # TODO: default agent options
    st.session_state.prompt_template_data1 = "" # TODO: default prompt template 1
    st.session_state.prompt_template_data2 = ""
    st.session_state.prompt_template_data3 = ""

    st.session_state.youling_server_id_value = ""
    st.session_state.task_id_value = ""
    st.session_state.use_example_files = False
    st.session_state.use_golden_samples = False
    st.session_state.use_multi_modal = False

@st.dialog("Prompt Template Management", width="large")
def prompt_template_management():
    prompt_template = st.text_area('Prompt Template 1', key=f'prompt_template', value=st.session_state.prompt_template_data1, height=150)
    prompt_template2 = st.text_area('Prompt Template 2 (Optional)', key=f'prompt_template2', placeholder="left empty if not needed", height=150, value=st.session_state.prompt_template_data2)
    prompt_template3 = st.text_area('Prompt Template 3 (Optional)', key=f'prompt_template3', placeholder="left empty if not needed", height=150, value=st.session_state.prompt_template_data3)
    
    col1, col2 = st.columns(2)
    if col1.button("Submit", use_container_width=True):
        st.session_state.prompt_template_data1 = prompt_template
        st.session_state.prompt_template_data2 = prompt_template2
        st.session_state.prompt_template_data3 = prompt_template3
        st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.rerun()

@st.dialog("Advanced Settings", width="large")
def advanced_settings():
    st.info("""If you want to use NetEase Youling Crowdsourcing Platform, you need to provide your server id. 
You can get your server id by following the doc [here](https://youling-platform.apps-hp.danlu.netease.com/docs/quickStart/annotation/stepsguide).
As an alternative, you can use the csv(human) agent to annotate the data.""")
    youling_server_id = st.text_input("Youling Server ID", key=f'youling_server_id', placeholder="56790", value=st.session_state.youling_server_id_value)
    st.session_state.youling_server_id_value = youling_server_id

    st.info("""The system will automatically create a unique task id for you (Recommended). Or, you can manually input a task id.""")
    task_id = st.text_input("Task ID", key=f'task_id', value=st.session_state.task_id_value)
    st.session_state.task_id_value = task_id

    st.info("The target confidence threshold for the task. Higher threshold makes the annotation more accurate, but may require more budget.")
    confidence_threshold = st.number_input('Confidence Threshold', key=f'confidence_threshold', min_value=0.0, max_value=1.0, value=0.99, step=0.01, help="The target confidence threshold for the task. Higher threshold makes the annotation more accurate, but may require more budget.")

    col1, col2 = st.columns(2)
    if col1.button("Submit", use_container_width=True):
        st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.rerun()



left1, right1 = st.columns([1,1])

with right1.container(border=True):
    task_description = st.text_area('Task Description', key=f'task_description', value=st.session_state.task_description_value, height=150)
    st.session_state.task_description_value = task_description

right1.html("""
<br>
""")

with right1.container(border=True):

    rl, rr = st.columns(2)

    rl.html("""
    <div style="height: 20px;"></div>
    """)
    
    with rl.popover("Use our example files", icon="💾", use_container_width=True):
        st.markdown("If you don't upload your data, system will use our [example files](https://drive.google.com/drive/folders/1DIk9pqm0Fl39mOuTjSuZ7LpaeXmNj_B9?usp=sharing) form the CrisisMMD dataset to run the annotation task as a demo. The task is to classify the disaster-related tweets into 4 categories.")

    options = ["Golden Samples", "Multi-modal"]
    selection = rr.pills("Directions", options, selection_mode="multi", label_visibility="hidden", default=["Golden Samples", "Multi-modal"])

    uploaded_unlabeled_file = st.file_uploader("Upload Unlabeled Data", key=f'file_unlabeled')
    if uploaded_unlabeled_file:
        dataframe = pd.read_csv(uploaded_unlabeled_file)
        train_file_path = f"./asserts/user_files/train.csv"
        dataframe.to_csv(train_file_path, index=False)
        
        
    if "Golden Samples" in selection:
        uploaded_labeled_file = st.file_uploader("Upload Golden Samples(Optional)", key=f'file_labeled')
        if uploaded_labeled_file:
            dataframe = pd.read_csv(uploaded_labeled_file)
            labeled_file_path = f"./asserts/user_files/labeled_gt.csv"
            dataframe.to_csv(labeled_file_path, index=False)
            st.session_state.use_golden_samples = True

    if "Multi-modal" in selection:
        images_zip_file = st.file_uploader("Upload Images Zip File(Optional)", key=f'file_images_zip')
        if images_zip_file:
            bytes_data = images_zip_file.read()
            images_zip_path = f"./asserts/user_files/images.zip"
            with open(images_zip_path, "wb") as f:
                f.write(bytes_data)
            st.session_state.use_multi_modal = True

with left1.container(border=True):
    left_top1, left_top2 = st.columns(2)
    with left_top1:
        task_name = st.text_input('Task Name', key=f'task_name', value=st.session_state.task_name_value)
        st.session_state.task_name_value = task_name
    with left_top2:
        budget_target = st.number_input('Budget Target ($)', key=f'budget', min_value=0, max_value=100000, value=st.session_state.budget_value, step=1, icon=":material/paid:")
        st.session_state.budget_value = budget_target

    st.html("""
    <br>
    """)

    custom_label  = st.multiselect(
        "Custom Label",
        ["positive", "negative"],
        default=st.session_state.custom_label_value,
        key=f'custom_label',
        accept_new_options=True,
    )
    

    agent_options = st.multiselect(
        "Select the annotation agents",
        ["gpt-4o-mini", "gpt-4o-mini(visual)", "MMBT", "RoBERTa", "CovNext V2", "NetEase Youling Crowdsourcing Platform", "csv(human)"],
        default=st.session_state.agent_options_value,
        key=f'agent_options'
    )
    

left1.html("""
<br>
""")


with left1.container(border=True):
    api_key = st.text_input('OpenAI API Key', key=f'openai_api_key', type='password', placeholder='sk-...', value=st.session_state.openai_api_key_value)
    st.session_state.openai_api_key_value = api_key
    base_url = st.text_input('OpenAI Base URL', key=f'openai_base_url', placeholder='https://api.openai.com/v1', value=st.session_state.openai_base_url_value)
    st.session_state.openai_base_url_value = base_url

left1.html("""
<br>
""")

left_left, left_right = left1.columns(2)
if left_left.button("Prompt Template", icon="📑", key=f'prompt_template_management_button', use_container_width=True):
    prompt_template_management()
if left_right.button("Advanced Settings", icon="🔧", key=f'advanced_settings_button', use_container_width=True):
    advanced_settings()

st.html("""
<br>
""")

@st.dialog("Confirm Task Configuration", width="large")
def confirm_task_configuration():
    if uploaded_unlabeled_file is None:
        st.session_state.use_example_files = True
    else:
        st.session_state.use_example_files = False

    st.session_state.custom_label_value = custom_label
    st.session_state.agent_options_value = agent_options
    if st.session_state.task_id_value == "":
        st.session_state.task_id_value = read_info.get_task_id(st.session_state.backend_path, st.session_state.task_name_value)

    if not st.session_state.task_submitted:
        read_info.submit_task(st.session_state)
        st.session_state.task_submitted = True

    confirm_col1, confirm_col2, confirm_col3 = st.columns(3)
    if confirm_col2.button("Go to Details", key="submit_confirm", use_container_width=True):
        st.session_state.task_submitted = False
        st.switch_page("pages/details.py")
        # st.rerun()


col1, col2, col3 = st.columns(3)
if col2.button("Submit", key="submit", use_container_width=True):
    flg, info = read_info.check_task_configuration(st.session_state)
    if flg:
        confirm_task_configuration()
    else:
        st.error(info)

