from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from ...state import State
from .prompt import SYSTEM_PROMPT


class ContactInfo(BaseModel):
    """Contact information for a guest"""

    name: str = Field(description="The name of the guest")
    email: str = Field(description="The email of the guest")
    phone: str = Field(description="The phone of the guest")
    age: str = Field(description="The age of the guest")


llm = init_chat_model("claude-haiku-4-5", temperature=0)
llm_with_structured_output = llm.with_structured_output(schema=ContactInfo)


def extractor(state: State):
    new_state: State = {}

    response_schema = llm_with_structured_output.invoke(
        [("system", SYSTEM_PROMPT), ("user", state["messages"])]
    )
    new_state["customer_name"] = response_schema.name
    new_state["phone"] = response_schema.phone
    new_state["my_age"] = response_schema.age

    return new_state
