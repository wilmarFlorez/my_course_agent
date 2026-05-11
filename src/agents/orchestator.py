import operator
from typing import Annotated, TypedDict, cast

from IPython.display import Image, Markdown, display
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
llm_worker = ChatAnthropic(model="claude-haiku-4-5", temperature=0, max_tokens=120)


class Section(BaseModel):
    name: str = Field(description="Name for this section of the report")
    description: str = Field(
        description=(
            "Brief overview of the main topics and "
            "concepts to be covered in this section"
        )
    )


class Sections(BaseModel):
    sections: list[Section] = Field(description="Sections of the report")


planner = llm.with_structured_output(Sections)


class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str


class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


def orchestrator(state: State):
    report_sections = cast(
        Sections,
        planner.invoke(
            [
                SystemMessage(content="Generate a plan for the report"),
                HumanMessage(content=f"Here is the report topic: {state['topic']}"),
            ]
        ),
    )
    return {"sections": report_sections.sections}


def llm_call(state: WorkerState):
    section = llm_worker.invoke(
        [
            SystemMessage(
                content=(
                    "Write a report section following the provided name and "
                    "description. Include no preambe. Use markdown formatting."
                )
            ),
            HumanMessage(
                content=f"""Section name: {state["section"].name}
Description: {state["section"].description}
"""
            ),
        ],
    )

    return {"completed_sections": [section.content]}


def synthesizer(state: State):
    return {"final_report": "\n\n---\n\n".join(state["completed_sections"])}


def assign_workers(state: State):
    return [Send("llm_call", {"section": s}) for s in state["sections"]]


builder = StateGraph(State)

builder.add_node("orchestator", orchestrator)
builder.add_node("llm_call", llm_call)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "orchestator")
builder.add_conditional_edges("orchestator", assign_workers, ["llm_call"])
builder.add_edge("llm_call", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()

