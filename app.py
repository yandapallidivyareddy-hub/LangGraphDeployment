import os
import sys
import io
import traceback
import uvicorn

from typing import TypedDict, Optional, Annotated

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langserve import add_routes

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ======================================
# GEMINI CONFIGURATION
# ======================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is not set."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ======================================
# STATE
# ======================================

class CrewState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    code: Optional[str]
    report: Optional[str]


# ======================================
# TOOL
# ======================================

@tool
def run_python_code(code: str) -> str:
    """
    Execute Python code and return output.
    """

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code.replace("```python", "")
            .replace("```", "")
            .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}

        exec(clean_code, {}, local_scope)

        result = new_stdout.getvalue()

    except Exception:
        result = traceback.format_exc()

    finally:
        sys.stdout = old_stdout

    return result if result else "Success (No Output)"


# ======================================
# DEVELOPER NODE
# ======================================

def developer(state: CrewState):

    task = state["messages"][-1].content

    prompt = f"""
You are an expert Python developer.

Task:
{task}

Rules:

- Return only executable Python code.
- Do not explain anything.
- Do not use markdown.
- Do not wrap the code inside triple backticks.
"""

    response = llm.invoke(prompt)

    return {
        "code": response.content
    }
# ======================================
# TESTER NODE
# ======================================

def tester(state: CrewState):

    code = state["code"]

    execution_result = run_python_code.invoke(
        {"code": code}
    )

    report = f"""
Generated Successfully

==============================
Execution Result
==============================

{execution_result}
"""

    return {
        "report": report
    }


# ======================================
# BUILD LANGGRAPH
# ======================================

builder = StateGraph(CrewState)

builder.add_node("developer", developer)
builder.add_node("tester", tester)

builder.add_edge(START, "developer")
builder.add_edge("developer", "tester")
builder.add_edge("tester", END)

langgraph_app = builder.compile()


# ======================================
# INPUT MODEL
# ======================================

class AgentInput(BaseModel):
    task: str = Field(
        description="Python coding task"
    )


# ======================================
# FORMAT INPUT
# ======================================

def format_input(x):

    task = x["task"] if isinstance(x, dict) else x.task

    return {
        "messages": [
            HumanMessage(content=task)
        ]
    }


# ======================================
# FORMAT OUTPUT
# ======================================

def format_output(state):

    return {
        "generated_code": state.get("code"),
        "report": state.get("report")
    }


# ======================================
# CREATE CHAIN
# ======================================

chain = (
    RunnableLambda(format_input)
    | langgraph_app
    | RunnableLambda(format_output)
).with_types(
    input_type=AgentInput
)
# ======================================
# FASTAPI APPLICATION
# ======================================

app = FastAPI(
    title="LangGraph Coding Agent",
    version="1.0.0"
)


# ======================================
# HEALTH CHECK
# ======================================

@app.get("/")
def health_check():
    return {
        "status": "LangGraph Coding Agent is running"
    }


# ======================================
# LANGSERVE ROUTES
# ======================================

add_routes(
    app,
    chain,
    path="/agent",
    playground_type="default"
)


# ======================================
# MAIN
# ======================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
