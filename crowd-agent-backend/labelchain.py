import logging
import time
import uuid
from typing import Annotated
from typing import Literal
from typing_extensions import TypedDict
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import InjectedState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
import support_function as support_function
import os
import copy
import traceback
import argparse


# TODO(optional): langsmith tracing
# os.environ['LANGSMITH_TRACING'] = 'true'
# os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGSMITH_API_KEY'] = 'xxxxxxxxxxxxxxxxxxxxxxxx'
# os.environ['LANGSMITH_PROJECT'] = 'xxxxxxxxxxxxxxxxxxxxxxxx'


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="mr_4k")
args = parser.parse_args()
TASK = args.task
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(ROOT_PATH, "output", TASK, "task_extra_info.yml")):
    os.remove(os.path.join(ROOT_PATH, "output", TASK, "task_extra_info.yml"))
if os.path.exists(os.path.join(ROOT_PATH, "output", TASK, "confusion_matrix.yml")):
    os.remove(os.path.join(ROOT_PATH, "output", TASK, "confusion_matrix.yml"))


support_function.update_status(TASK, ROOT_PATH, "Running")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
for handler in rootLogger.handlers[:]:
    rootLogger.removeHandler(handler)
rootLogger.setLevel(logging.INFO)
lctime = time.localtime()
lctime = time.strftime("%Y-%m-%d_%A_%H:%M:%S",lctime)
log_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output",
    TASK,
    f"{lctime}.log"
)
fileHandler = logging.FileHandler(log_path)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)


class SimpleState(TypedDict):
    messages: Annotated[list[str], add]
    task: str
    train_path: str
    dev_path: str
    project_root: str
    loop_count: int
    last_selected_samples_path: str
    last_data_embeddings_path: str
    left_cnt: int
    budget: int
    confidence_threshold: float
    agent_name_list: list[str]
    cost_role: str
    evaluate_role: str
    id_to_label: dict
    quality_chain_info: str
    chain_role: str
    financial_chain_info: str
    rules_path: str
    profile_role: str

@tool
def human_annotate_tool(state: Annotated[dict, InjectedState]):
    """Request human annotation data, accurate but expensive, expected to consume about 3$ per round, annotate about 200 samples."""
    tool_call_id = state.get("messages")[-1].tool_calls[0].get("id")

    mode = support_function.human_annotate(state, online=True)

    return Command(
        update={
            "messages": [ToolMessage(f"Annotation Agent: Human annotation completed. Using {mode} mode.", tool_call_id=tool_call_id)],
            "agent_name_list": ["human"],
            "cost_role": "human",
            "evaluate_role": "human",
            "chain_role": "human",
            "profile_role": "human_annotate_tool",
        }
    )


@tool
def llm_annotate_tool(state: Annotated[dict, InjectedState]):
    """Request text LLM annotation data, cheap, accuracy is general, will learn the correct annotation results in subsequent rounds (annotation accuracy improvement), expected to consume less than 0.2$ per round"""
    tool_call_id = state.get("messages")[-1].tool_calls[0].get("id")

    try:
        if  state.get("loop_count") == 1:
            support_function.update_extra_info(state, {"agent_name": "gpt-4o-mini", "label_count": state.get("left_cnt")})
            agent_name_list = ["llm_simple"] # ["llm_tf","llm_choice", "llm_simple", "llm_swapping", "llm_bias"]
            support_function.llm_first_annotate(state, agent_name_list)
            role = "llm"
        else:
            support_function.llm_refine_annotate(state)
            agent_name_list = ["text_llm_refine"]
            role = "text_llm_refine"
    except Exception as e:
        print(f"Error in llm_annotate_tool: {e}")
        print(traceback.format_exc())
        raise e
    return Command(
        update={
            "messages": [ToolMessage("Annotation Agent: gpt-4o-mini annotation done.", tool_call_id=tool_call_id)],
            "agent_name_list": agent_name_list,
            "cost_role": "gpt-4o-mini",
            "evaluate_role": role,
            "chain_role": "gpt-4o-mini",
            "profile_role": "llm_annotate_tool",
        }
    )


@tool
def vlm_annotate_tool(state: Annotated[dict, InjectedState]):
    """Request visual language large model annotation data, can use visual information to answer questions, accuracy is better than text large model, will learn the correct annotation results in subsequent rounds (annotation accuracy improvement), expected to consume less than 0.7$ per round, annotate about 400 samples."""
    tool_call_id = state.get("messages")[-1].tool_calls[0].get("id")

    if state.get("loop_count") == 1:
        agent_name_list = ["visual_llm_simple"] # ["llm_tf","llm_choice", "llm_simple", "llm_swapping", "llm_bias"]
        support_function.llm_first_annotate(state, agent_name_list)
        role = "visual_llm"
    else: 
        support_function.llm_refine_annotate(state, visual=True)
        agent_name_list = ["visual_llm_refine"]
        role = "visual_llm_refine"

    return Command(
        update={
            "messages": [ToolMessage("Annotation Agent: Visual language large model annotation completed.", tool_call_id=tool_call_id)],
            "agent_name_list": agent_name_list,
            "cost_role": "gpt-4o-mini",
            "evaluate_role": role,
            "chain_role": "visual large model",
            "profile_role": "vlm_annotate_tool",
        }
    )


@tool
def text_slm_annotate_tool(state: Annotated[dict, InjectedState]):
    """Request text small model annotation data, use text information to annotate, cost and accuracy are between large model and human, expected to consume less than 0.2$ per round"""
    tool_call_id = state.get("messages")[-1].tool_calls[0].get("id")

    support_function.update_extra_info(state, {"agent_name": "RoBERTa", "label_count": state.get("left_cnt")})

    support_function.slm_train(state, type="roberta")
    selected_sample_path, data_embeddings_path = support_function.slm_fliter_samples(state, type="roberta")
    agent_name_list = ["roberta_high_confidence", "roberta_other_confidence"]

    return Command(
        update={
            "messages": [ToolMessage("Annotation Agent: RoBERTa annotation completed.", tool_call_id=tool_call_id)],
            "agent_name_list": agent_name_list,
            "cost_role": "1080ti",
            "evaluate_role": "roberta",
            "last_selected_samples_path": selected_sample_path,
            "last_data_embeddings_path": data_embeddings_path,
            "chain_role": "RoBERTa",
            "profile_role": "text_slm_annotate_tool",
        }
    )


@tool
def visual_slm_annotate_tool(state: Annotated[dict, InjectedState]):
    """Request visual small model annotation data, use visual information to annotate, cost and accuracy are between large model and human, expected to consume less than 0.2$ per round"""
    tool_call_id = state.get("messages")[-1].tool_calls[0].get("id")

    support_function.update_extra_info(state, {"agent_name": "ConvNext v2", "label_count": state.get("left_cnt")})

    support_function.slm_train(state, type="convnextv2")
    selected_sample_path, data_embeddings_path = support_function.slm_fliter_samples(state, type="convnextv2")
    agent_name_list = ["convnextv2_high_confidence", "convnextv2_other_confidence"]

    return Command(
        update={
            "messages": [ToolMessage("Annotation Agent: ConvNext V2 annotation completed.", tool_call_id=tool_call_id)],
            "agent_name_list": agent_name_list,
            "cost_role": "1080ti",
            "evaluate_role": "convnextv2",
            "last_selected_samples_path": selected_sample_path,
            "last_data_embeddings_path": data_embeddings_path,
            "chain_role": "ConvNext V2",
            "profile_role": "visual_slm_annotate_tool",
        }
    )


@tool
def multi_modal_slm_annotate_tool(state: Annotated[dict, InjectedState]):
    """Request multi-modal small model annotation data, use both text and visual information to annotate, cost and accuracy are between large model and human, expected to consume less than 0.2$ per round"""
    tool_call_id = state.get("messages")[-1].tool_calls[0].get("id")

    support_function.update_extra_info(state, {"agent_name": "MMBT", "label_count": state.get("left_cnt")})

    support_function.slm_train(state, type="mmbt")
    selected_sample_path, data_embeddings_path = support_function.slm_fliter_samples(state, type="mmbt")
    agent_name_list = ["mmbt_high_confidence", "mmbt_other_confidence"]

    return Command(
        update={
            "messages": [ToolMessage("Annotation Agent: MMBT annotation completed.", tool_call_id=tool_call_id)],
            "agent_name_list": agent_name_list,
            "cost_role": "1080ti",
            "evaluate_role": "mmbt",
            "last_selected_samples_path": selected_sample_path,
            "last_data_embeddings_path": data_embeddings_path,
            "chain_role": "MMBT",
            "profile_role": "multi_modal_slm_annotate_tool",
        }
    )


def init_pipeline(state: SimpleState):
    return {
        "loop_count": 0,
        "project_root": ROOT_PATH,
        "task": TASK,
        "train_path": f"{ROOT_PATH}/data/{TASK}/train.csv",
        "dev_path": f"{ROOT_PATH}/data/{TASK}/labeled_gt.csv",
        "left_cnt": support_function.get_left_cnt(f"{ROOT_PATH}/data/{TASK}/train.csv"),
        "budget": support_function.get_budget(TASK, ROOT_PATH),
        "confidence_threshold": 0.99,
        "last_data_embeddings_path": "",
        "id_to_label": support_function.get_detail_labels(TASK, ROOT_PATH),
        "quality_chain_info": "call path(agent, accuracy, unconverged count):",
        "financial_chain_info": "call path(cost, remaining budget):",
        "rules_path": "",
    }


def wrong_samples_review(state: SimpleState):
    wrong_samples = support_function.get_wrong_samples(state)
    if len(wrong_samples) > 0:
        wrong_samples_analysis, rules_path, res = support_function.get_wrong_samples_analysis(state, wrong_samples)
        res.content = "Wrong Sample Analysis: " + res.content


        update_messages = [res]
        if res.tool_calls:
            update_messages.append(
                ToolMessage(
                    content=f"Annotation Rules Saved.",
                    tool_call_id=res.tool_calls[0].get("id")
                )
            )

        return Command(
            update={
                "messages": update_messages,
                "rules_path": rules_path,
            }
        )



def quality_review(state: SimpleState):

    support_function.get_confusion_matrix_on_labeled_samples(state)  # agent_name_list
    support_function.tag_aggregation(state)
    confidence_distribution, left_cnt = support_function.get_confidence_distribution(state)
    count_labeled_samples_and_acc, acc = support_function.evaluate_on_labeled_samples(state)  # evaluate_role
    quality_chain_info = support_function.update_quality_chain_info(state, acc, left_cnt)  # chain_role
    support_function.update_extra_info(state, {"accuracy": acc})
    response_message = ""

    messages = state.get("messages")
    temp_messages = copy.deepcopy(messages)
    temp_messages.append(
        SystemMessage(
            content=support_function.get_quality_review_instruction(confidence_distribution, count_labeled_samples_and_acc, quality_chain_info)
        )
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key="xxxxxxxxxxxxxxxxxxxxx",  # TODO: config your own api key
        base_url="https://api.openai.com/v1", 
    )
    ai_message = llm.invoke(temp_messages)
    response_message = ai_message.content

    return Command(
        update={
            "messages": [
                AIMessage(
                    content=f"QA Agent: {confidence_distribution}\n{count_labeled_samples_and_acc}\n{quality_chain_info}\n{response_message}",
                )
            ],
            "left_cnt": left_cnt,
            "quality_chain_info": quality_chain_info,
        }
    )


def update_user_profile(state: SimpleState):
    confusion_matrix_readable = support_function.read_confusion_matrix(state)
    user_profile_analysis, profile, res = support_function.get_user_profile(state, confusion_matrix_readable)

    res.content = "Annotator Profile Update: " + res.content

    update_messages = [res]
    if res.tool_calls:
        update_messages.append(
            ToolMessage(
                content=f"Annotator Profile Saved.",
                tool_call_id=res.tool_calls[0].get("id")
            )
        )
    return Command(
        update={
            "messages": update_messages,
        }
    )

def financial_review(state: SimpleState):
    now_budget = state.get("budget")
    cost = support_function.get_cost(state)  # cost_role
    financial_chain_info = support_function.update_financial_chain_info(state, cost, now_budget-cost)  # cost_role
    support_function.update_extra_info(state, {"budget_cost": cost})
    response_message = ""
    
    messages = state.get("messages")
    temp_messages = copy.deepcopy(messages)
    temp_messages.append(
        SystemMessage(
            content=support_function.get_financial_review_instruction(cost, now_budget-cost, financial_chain_info)
        )
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key="xxxxxxxxxxxxxxxxxxxxx",  # TODO: config your own api key
        base_url="https://api.openai.com/v1", 
    )
    ai_message = llm.invoke(temp_messages)
    response_message = ai_message.content

    return Command(
        update={
            "messages": [
                AIMessage(
                    content=f"Financing Agent: This round cost {cost}$, remaining budget {now_budget-cost}$.\n{financial_chain_info}\n{response_message}",
                )
            ],
            "budget": now_budget - cost,
            "financial_chain_info": financial_chain_info,
        }
    )


def report(state: SimpleState):
    pass


def choose_tool(name: str, now_loop: int) -> AIMessage:
    message_text = "Scheduling Agent: This round is managed by preset rules, skipping the analysis step."


    return AIMessage(
            content=message_text,
            tool_calls=[
                {
                    "name": name,
                    "args": {},
                    "id": "tool_call_id",
                    "type": "tool_call",
                }
            ],
        )


def loop_judge(
    state: SimpleState,
) -> Command[Literal["report", "check_tool_called"]]:
    now_loop = state["loop_count"]
    left_cnt = state["left_cnt"]

    router_dict = {
        0: "llm_annotate_tool",
    }


    if left_cnt == 0 or now_loop > 20:
        return Command(
            goto="report",
        )
    else:
        messages = state.get("messages")
        temp_messages = copy.deepcopy(messages)
        last_message = temp_messages[-1]
        if "No tool was called" not in last_message.content:
            temp_messages.append(
                SystemMessage(
                    content=support_function.get_annotator_profiles(state)
                )
            )
            temp_messages.append(
                SystemMessage(
                    content=support_function.get_planner_instruction()
                )
            )
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key="xxxxxxxxxxxxxxxxxxxxx",  # TODO: config your own api key
            base_url="https://api.openai.com/v1", 
        )
        model_with_tools = llm.bind_tools(tools)
        ai_message = model_with_tools.invoke(temp_messages)
        ai_message.content = "Scheduling Agent: " + ai_message.content
        return Command(
            update={
                "loop_count": now_loop + 1,
                "messages": [ai_message],
            }, goto="check_tool_called"
        )
    
def check_tool_called(
    state: SimpleState,
) -> Command[Literal["planner", "annotation_agents"]]:
    messages = state.get("messages")
    now_loop = state["loop_count"]
    tool_message = messages[-1]
    if not tool_message.tool_calls:
        return Command(
            update={
                "messages": [
                    SystemMessage(
                        content="No tool was called. Please call a tool to annotate the data."
                    )
                ],
                "loop_count": now_loop - 1,
            }, goto="planner"
        )
    
    return Command(
        goto="annotation_agents",
    )


graph_builder = StateGraph(SimpleState)
agents = support_function.get_annotation_agents(TASK, ROOT_PATH)
agent_mapping_dict = {
    "csv(human)": human_annotate_tool,
    "gpt-4o-mini": llm_annotate_tool,
    "RoBERTa": text_slm_annotate_tool,
    "MMBT": multi_modal_slm_annotate_tool,
    "gpt-4o-mini(visual)": vlm_annotate_tool,
    "CovNext V2": visual_slm_annotate_tool,
}
tools = []
for agent in agents:
    if agent in agent_mapping_dict:
        tools.append(agent_mapping_dict[agent])
    else:
        raise ValueError(f"Agent {agent} not found in agent_mapping_dict. Please check the agent name.")

annotation_agents = ToolNode(tools)

graph_builder.add_node("init", init_pipeline)
graph_builder.add_node("planner", loop_judge)
graph_builder.add_node("check_tool_called", check_tool_called)
graph_builder.add_node("annotation_agents", annotation_agents)
graph_builder.add_node("report", report)
graph_builder.add_node("wrong_samples_review", wrong_samples_review)
graph_builder.add_node("quality_review", quality_review)
graph_builder.add_node("update_user_profile", update_user_profile)
graph_builder.add_node("financial_review", financial_review)

graph_builder.add_edge(START, "init")
graph_builder.add_edge("init", "planner")
graph_builder.add_edge("annotation_agents", "wrong_samples_review")
graph_builder.add_edge("wrong_samples_review", "quality_review")
graph_builder.add_edge("quality_review", "update_user_profile")
graph_builder.add_edge("update_user_profile", "financial_review")
graph_builder.add_edge("financial_review", "planner")
graph_builder.add_edge("report", END)


checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)
thread_config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
    },
    "recursion_limit": 100,
}

last_chunk = None

for chunk in graph.stream(
    {
        "messages": [
            HumanMessage(content=support_function.get_human_request(TASK, ROOT_PATH)),
        ],
    }, stream_mode="values",config=thread_config,
):
    chunk["messages"][-1].pretty_print()
    logging.info(chunk)
    support_function.save_chunk(chunk, TASK, ROOT_PATH)
    last_chunk = chunk


support_function.update_status(TASK, ROOT_PATH, "Finished")

if last_chunk is not None:
    if "messages" in last_chunk:
        print("\n\n===============last_chunk['messages']===============")
        for message in last_chunk['messages']:
            print(message.content)
            print("\n")


mermaid_image = graph.get_graph().draw_mermaid_png()
with open("labelchain.png", "wb") as f:
    f.write(mermaid_image)