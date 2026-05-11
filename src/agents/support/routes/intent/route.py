from typing import Literal, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from agents.support.routes.intent.prompt import SYSTEM_PROMPT
from agents.support.state import State


class RouteIntent(BaseModel):
    """Contact information for a person."""

    step: Literal["conversation", "booking"] = Field(
        description="The next step in the routing process"
    )


llm = init_chat_model("gpt-5.4-mini", temperature=0)
llm_with_structured_output = llm.with_structured_output(schema=RouteIntent)


def intent_route(state: State) -> Literal["conversation", "booking"]:
    history = state["messages"]
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *history
    ]
    schema = cast(
        RouteIntent, llm_with_structured_output.invoke(messages)
    )
    return schema.step
