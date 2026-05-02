from random import randint

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, MessagesState, StateGraph

llm = init_chat_model("gpt-5.4-nano", temperature=1)
file_search_tool = {
    "type": "file_search",
    "vector_store_ids": ["vs_69f5149726ac8191acac9ac2f4d5df7f"],
}

llm_with_tools = llm.bind_tools([file_search_tool])


class State(MessagesState):
    customer_name: str
    my_age: int


def node_1(state: State):
    new_state: State = {}
    if state.get("customer_name") is None:
        new_state["customer_name"] = "Wilmar Florez"
    else:
        new_state["my_age"] = randint(18, 90)

    history = state["messages"]
    last_message = history[-1]
    ai_message = llm_with_tools.invoke(last_message.text)
    new_state["messages"] = [ai_message]
    print(new_state)
    return new_state


builder = StateGraph(State)
builder.add_node("node_1", node_1)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

agent = builder.compile()
