from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from typing import List, Literal
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Optional
import yaml
import logging
import traceback

def ask_open_ai(messages):
    with open("config/api_key.yml", "r") as f:
        config = yaml.safe_load(f)

    open_ai_api_key = config["open_ai"]["api_key"]
    open_ai_base_url = (
        config["open_ai"]["base_url"] if "base_url" in config["open_ai"] else None
    )
    model_name = config["open_ai"]["model_name"]
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=open_ai_api_key,
        base_url=open_ai_base_url,
    )
    return llm.invoke(messages)


def ask_open_ai_with_callback(messages):
    with open("config/api_key.yml", "r") as f:
        config = yaml.safe_load(f)

    open_ai_api_key = config["open_ai"]["api_key"]
    open_ai_base_url = (
        config["open_ai"]["base_url"] if "base_url" in config["open_ai"] else None
    )
    model_name = config["open_ai"]["model_name"]
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=open_ai_api_key,
        base_url=open_ai_base_url,
    )

    with get_openai_callback() as cb:
        res = llm.invoke(messages)
    return res, cb


def ask_open_ai_with_structured_output(messages, category_to_label_dict):
    with open("config/api_key.yml", "r") as f:
        config = yaml.safe_load(f)

    open_ai_api_key = config["open_ai"]["api_key"]
    open_ai_base_url = (
        config["open_ai"]["base_url"] if "base_url" in config["open_ai"] else None
    )
    model_name = config["open_ai"]["model_name"]
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=open_ai_api_key,
        base_url=open_ai_base_url,
    )
    CLASSIFICATION_CATEGORYS = list(category_to_label_dict.keys())
    class ClassificationTask(BaseModel):
        """ classification task """
        category: Literal[tuple(CLASSIFICATION_CATEGORYS)] = Field(description="The category to classify the text")

    model_with_tools = llm.bind_tools([ClassificationTask])
    with get_openai_callback() as cb:
        res = model_with_tools.invoke(messages)

    pydantic_object = ClassificationTask.model_validate(res.tool_calls[-1]["args"])
    label = category_to_label_dict[pydantic_object.category]
    return res, cb, pydantic_object.category, label


def ask_open_ai_wrong_samples_review(messages):
    with open("config/api_key.yml", "r") as f:
        config = yaml.safe_load(f)

    open_ai_api_key = config["open_ai"]["api_key"]
    open_ai_base_url = (
        config["open_ai"]["base_url"] if "base_url" in config["open_ai"] else None
    )
    model_name = config["open_ai"]["model_name"]
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=open_ai_api_key,
        base_url=open_ai_base_url,
    )
    class SaveAnnotationRules(BaseModel):
        """ save annotation rules """
        rules: str = Field(description="pass the detailed annotation rules as an arg")

    model_with_tools = llm.bind_tools([SaveAnnotationRules])
    with get_openai_callback() as cb:
        res = model_with_tools.invoke(messages)

    rules = ""
    try:
        pydantic_object = SaveAnnotationRules.model_validate(res.tool_calls[-1]["args"])
        rules = pydantic_object.rules
    except Exception as e:
        rules = res.content
    return res, cb, rules


def ask_open_ai_generate_user_profile(messages):
    with open("config/api_key.yml", "r") as f:
        config = yaml.safe_load(f)

    open_ai_api_key = config["open_ai"]["api_key"]
    open_ai_base_url = (
        config["open_ai"]["base_url"] if "base_url" in config["open_ai"] else None
    )
    model_name = config["open_ai"]["model_name"]
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=open_ai_api_key,
        base_url=open_ai_base_url,
    )
    class SaveCapabilityProfile(BaseModel):
        """save generated capability profile"""
        profile: str = Field(description="the generated capability profile")

    model_with_tools = llm.bind_tools([SaveCapabilityProfile])
    with get_openai_callback() as cb:
        res = model_with_tools.invoke(messages)
    
    profile = ""
    try:
        pydantic_object = SaveCapabilityProfile.model_validate(res.tool_calls[-1]["args"])
        profile = pydantic_object.profile
    except Exception as e:
        profile = res.content
    return res, cb, profile


if __name__ == "__main__":
    print(ask_open_ai("hello"))
