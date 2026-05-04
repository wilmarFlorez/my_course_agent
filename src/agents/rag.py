from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

llm = init_chat_model("gpt-5.4-nano", temperature=1)
file_search_tool = {
    "type": "file_search",
    "vector_store_ids": ["vs_69f5149726ac8191acac9ac2f4d5df7f"],
}

llm_with_tools = llm.bind_tools([file_search_tool])


class State(MessagesState):
    customer_name: str
    my_age: str
    phone: str


class ContactInfo(BaseModel):
    """Contact information for a guest"""

    name: str = Field(description="The name of the guest")
    email: str = Field(description="The email address of the guest")
    phone: str = Field(description="The phone number of the person")
    age: str = Field(description="The age of the guest")


llm_claude = init_chat_model("claude-haiku-4-5", temperature=0)
llm_with_structured_output = llm_claude.with_structured_output(schema=ContactInfo)


def extractor(state: State):
    new_state: State = {}

    response_schema = llm_with_structured_output.invoke(state["messages"])
    new_state["customer_name"] = response_schema.name
    new_state["phone"] = response_schema.phone
    new_state["my_age"] = response_schema.age

    return new_state


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


builder = StateGraph(State)
builder.add_node("extractor", extractor)
builder.add_node("conversation", conversation)

builder.add_edge(START, "extractor")
builder.add_edge("extractor", "conversation")
builder.add_edge("conversation", END)

agent = builder.compile()
