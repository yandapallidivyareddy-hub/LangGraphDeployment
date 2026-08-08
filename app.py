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


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is not set."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# STATE
# ============================================================

class CrewState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    code: Optional[str]
    execution_result: Optional[str]
    report: Optional[str]


# ============================================================
# PYTHON CODE EXECUTION TOOL
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """
    Execute generated Python code and return the output.
    """

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    # --------------------------------------------------------
    # Basic safety restrictions
    # --------------------------------------------------------

    forbidden = [
        "import os",
        "import sys",
        "import subprocess",
        "import socket",
        "import shutil",
        "import pathlib",
        "from os",
        "from sys",
        "from subprocess",
        "open(",
        "__import__",
        "eval(",
        "exec(",
        "compile(",
        "globals(",
        "locals(",
        "getattr(",
        "setattr(",
        "delattr(",
    ]

    for item in forbidden:
        if item in clean_code:
            return f"Execution blocked: use of '{item}' is not allowed."

    # --------------------------------------------------------
    # Capture stdout
    # --------------------------------------------------------

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:
        safe_builtins = {
            "print": print,
            "len": len,
            "range": range,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "reversed": reversed,
            "bool": bool,
        }

        safe_globals = {
            "__builtins__": safe_builtins
        }

        safe_locals = {}

        exec(
            clean_code,
            safe_globals,
            safe_locals
        )

        result = new_stdout.getvalue()

    except Exception:
        result = traceback.format_exc()

    finally:
        sys.stdout = old_stdout

    if result.strip():
        return result.strip()

    return "Code executed successfully with no output."


# ============================================================
# DEVELOPER NODE
# ============================================================

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
- Make sure the program produces visible output when appropriate.
"""

    response = llm.invoke(prompt)

    generated_code = response.content

    if isinstance(generated_code, list):
        generated_code = "".join(
            item.get("text", "")
            for item in generated_code
            if isinstance(item, dict)
        )

    return {
        "code": generated_code
    }


# ============================================================
# TESTER NODE
# ============================================================

def tester(state: CrewState):

    code = state.get("code", "")

    execution_result = run_python_code.invoke(
        {"code": code}
    )

    report = (
        "Generated Successfully\n\n"
        "Execution Result:\n"
        f"{execution_result}"
    )

    return {
        "execution_result": execution_result,
        "report": report
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(CrewState)

builder.add_node("developer", developer)
builder.add_node("tester", tester)

builder.add_edge(START, "developer")
builder.add_edge("developer", "tester")
builder.add_edge("tester", END)

langgraph_app = builder.compile()


# ============================================================
# INPUT MODEL
# ============================================================

class AgentInput(BaseModel):
    task: str = Field(
        description="Python coding task"
    )


# ============================================================
# FORMAT INPUT
# ============================================================

def format_input(x):

    if isinstance(x, dict):
        task = x["task"]
    else:
        task = x.task

    return {
        "messages": [
            HumanMessage(content=task)
        ]
    }


# ============================================================
# FORMAT OUTPUT
# ============================================================

def format_output(state):

    return {
        "generated_code": state.get("code", ""),
        "execution_result": state.get(
            "execution_result",
            ""
        ),
        "report": state.get(
            "report",
            ""
        )
    }


# ============================================================
# CREATE CHAIN
# ============================================================

chain = (
    RunnableLambda(format_input)
    | langgraph_app
    | RunnableLambda(format_output)
).with_types(
    input_type=AgentInput
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="LangGraph Coding Agent",
    version="1.0.0"
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health_check():

    return {
        "status": "LangGraph Coding Agent is running"
    }


# ============================================================
# LANGSERVE ROUTES
# ============================================================

add_routes(
    app,
    chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
