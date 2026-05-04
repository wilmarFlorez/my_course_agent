from langchain.chat_models import init_chat_model

from ...state import State

llm = init_chat_model("gpt-5.4-nano", temperature=0)

file_search_tool = {
    "type": "file_search",
    "vector_store_ids": ["vs_69f5149726ac8191acac9ac2f4d5df7f"],
}

llm_with_tools = llm.bind_tools([file_search_tool])


def conversation(state: State):
    new_state: State = {}
    history = state["messages"]
    last_message = history[-1]
    customer_name = state.get("customer_name", "Jhon Doe")
    system_message = f"""You are a helpful asistant that can answer questions about the 
    custoemr {customer_name}"""
    ai_message = llm_with_tools.invoke(
        [("system", system_message), ("user", last_message.text)]
    )
    new_state["messages"] = [ai_message]
    print(new_state)
    return new_state
