from langchain.chat_models import init_chat_model

from src.agents.support.nodes.extractor.prompt import SYSTEM_PROMPT

from ...state import State
from .tools import tools

llm = init_chat_model("gpt-5.4-nano", temperature=0)


llm_with_tools = llm.bind_tools(tools)


def conversation(state: State):
    new_state: State = {}
    history = state["messages"]
    last_message = history[-1]
    customer_name = state.get("customer_name", "Jhon Doe")
    system_message = f"""{SYSTEM_PROMPT} {customer_name}"""
    ai_message = llm_with_tools.invoke(
        [("system", system_message), ("user", last_message.text)]
    )
    new_state["messages"] = [ai_message]
    print(new_state)
    return new_state
