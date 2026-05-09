from langchain.agents import create_agent

from agents.support.nodes.booking.prompt import prompt
from agents.support.nodes.booking.tools import tools

booking_node = create_agent(
    model="gpt-5-mini", tools=tools, system_prompt=prompt
)
