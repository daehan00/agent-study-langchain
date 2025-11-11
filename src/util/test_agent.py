from typing import Annotated
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import TodoListMiddleware
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

from pydantic import BaseModel, Field
from .ai_models import llm_small, llm


#-------------------------
@tool
def read_file(file_path: str) -> str:
    """Read contents of a file."""
    with open(file_path) as f:
        return f.read()


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file."""
    with open(file_path, 'w') as f:
        f.write(content)
    return f"Wrote {len(content)} characters to {file_path}"


@tool
def run_tests(test_path: str) -> str:
    """Run tests and return results."""
    # Simplified for example
    return "All tests passed!"


todo_agent = create_agent(
    model=llm_small,
    tools=[read_file, write_file, run_tests],
    middleware=[TodoListMiddleware()],
)
#-------------------------
class TravelInfo(BaseModel):
    destination: str = Field(
        description="여행 목적지입니다."
    )
    information: str = Field(
        description="먹거리, 명소, 역사 등 여행지에 대한 정보입니다."
    )

class State(AgentState):
    destination: str



@tool
def tool_2(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """여행 계획의 개요를 작성하는 도구입니다.
    
    Args:
        query: str = 여행지와 여행 목적
    Return:
        Command: 여행 계획 개요"""
    
    
    response = llm_small.invoke(f"다음의 여행지와 여행 목적에 맞도록 여행 계획 개요 항목을 작성해 주세요.\n{query}")

    return Command(update={
        "messages": [
            ToolMessage(
                content=str(response.content),
                tool_call_id=tool_call_id,
                name="tool_2"
            )
        ]
    })

@tool
def tool_3(query: str) -> str:
    """여행지에 대한 정보를 수집하는 도구입니다.
    
    Args:
        query: str = 여행지
    Return:
        str: 여행지 정보"""

    response = llm_small.invoke([
        SystemMessage(content="당신은 한국의 여행지 전문가입니다."),
        HumanMessage(content=f"{query}의 먹거리, 명소, 역사 등 정보를 알려주세요.")
    ])

    return str(response.content)

tour_agent = create_agent(
    model=llm,
    tools=[tool_2, tool_3],
    state_schema=State,
    system_prompt="당신은 한국의 여행사 직원입니다. 도구를 활용하여 고객의 여행 계획을 수립하세요." \
    "계획 수립에 앞서 먼저 여행지에 대한 조사를 수행하세요.",
)
#-------------------------
@tool
def plan_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """여행 계획의 개요를 작성하는 도구입니다.
    
    Args:
        query: str = 여행지와 여행 목적
    Return:
        Command: 여행 계획 개요"""
    
    response = llm_small.invoke(f"다음의 여행지와 여행 목적에 맞도록 여행 계획 개요 항목을 작성해 주세요.\n{query}")

    return Command(update={
        "messages": [
            ToolMessage(
                content=str(response.content),
                tool_call_id=tool_call_id,
                name="plan_tool"
            ), response
        ]
    })

@tool
def info_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """여행지에 대한 정보를 수집하는 도구입니다.
    
    Args:
        query: str = 여행지
    Return:
        str: 여행지 정보"""
    response = tour_agent.invoke({
        "messages":[
            SystemMessage(content="당신은 한국의 여행지 전문가입니다."),
            HumanMessage(content=f"{query}의 먹거리, 명소, 역사 등 정보를 알려주세요.")
        ]
    })

    return Command(update={
        "messages": [
            ToolMessage(
                content=str(response["messages"][-1].content),
                tool_call_id=tool_call_id,
                name="info_tool"
            )
        ] + response["messages"][2:]
    })