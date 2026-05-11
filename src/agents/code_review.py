from typing import TypedDict, cast

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

llm = init_chat_model("gpt-5.4-mini", temperature=0)


class SecurityReview(BaseModel):
    vulnerabilities: list[str] = Field(
        description="The vulnerabilities in the code", default=[]
    )
    riskLevel: str = Field(
        description="The risk level of the vulnerabilities", default=""
    )
    suggestions: list[str] = Field(
        description="The suggestions for fixing the vulnerabilities", default=[]
    )


class MaintainabilityReview(BaseModel):
    concerns: list[str] = Field(description="The concerns about the code", default=[])
    qualityScore: int = Field(
        description="The quality score of the code from 1 to 10", default=1, ge=1, le=10
    )


class State(TypedDict):
    code: str
    security_review: SecurityReview
    maintainability_review: MaintainabilityReview
    final_review: str


def security_review(state: State):
    code = state["code"]
    messages = [
        (
            "system",
            """You are an expert in code security. Focus on identifiying security
        vulnerabilities, injection risks, and authentication issues.""",
        ),
        ("user", f"Review tis code: {code}"),
    ]

    llm_with_structured_output = llm.with_structured_output(SecurityReview)
    result = cast(SecurityReview, llm_with_structured_output.invoke(messages))

    return {"security_review": result}


def maintainability_review(state: State):
    code = state["code"]
    messages = [
        (
            "system",
            """You are an expert in code quality. Focus on code structure, readability, 
            and adherence to best practices.""",
        ),
        ("user", f"Review tis code: {code}"),
    ]

    llm_with_structured_output = llm.with_structured_output(MaintainabilityReview)
    result = cast(
        MaintainabilityReview, llm_with_structured_output.invoke(messages)
    )


    return {"maintainability_review": result}


def aggregator(state: State):
    security_review = state["security_review"]
    maintainability_review = state["maintainability_review"]

    messages = [
        ("system", "You are a technical lead summarizing multiple code reviews"),
        (
            "user",
            f"""Synthesize these code review results into a concise summary with 
            key actions: Security review: {security_review} and Maintainability 
            review: {maintainability_review}""",
        ),
    ]

    response = llm.invoke(messages)
    print('RESPONSE ==>', response)
    return {
        'final_review':response.content
    }


builder = StateGraph(State)

builder.add_node("security_review", security_review)
builder.add_node("maintainability_review", maintainability_review)
builder.add_node("aggregator", aggregator)

builder.add_edge(START, "security_review")
builder.add_edge(START, "maintainability_review")
builder.add_edge("security_review", "aggregator")
builder.add_edge("maintainability_review", "aggregator")
builder.add_edge("aggregator", END)
agent = builder.compile()

