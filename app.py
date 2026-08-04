import sys
import io
import traceback
import os
import uvicorn
import asyncio
from langchain_core.tools import tool
from fastapi import FastAPI
from pydantic import BaseModel, Field
import nest_asyncio
nest_asyncio.apply()

from langserve import add_routes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ======================================
# GEMINI CONFIGURATION
# ======================================

try:
    # Try to get from environment first, then from userdata
    # FIXED: Ensured variable naming consistency to avoid NameError
    GOOGLE_API_KEY = os.environ.get("GOOGLE_APIKEY") or userdata.get('GOOGLE_APIKEY')
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
except Exception as e:
    print(f"Warning: Could not retrieve GOOGLE_APIKEY: {e}")
    GOOGLE_API_KEY = ""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
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

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return output."""
    if not isinstance(code, str):
        code = str(code)
    clean_code = code.replace("```python", "").replace("```", "").strip()
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
# NODES
# ======================================

def developer(state: CrewState):
    task = state["messages"][-1].content
    prompt = f"You are an expert Python developer.\n\nTask:\n{task}\n\nRules:\n- Return only executable Python code.\n- Do not explain anything.\n- Do not use markdown.\n- Do not wrap the code in ``` blocks."
    response = llm.invoke(prompt)
    return {"code": response.content}

def tester(state: CrewState):
    code = state["code"]
    execution_result = run_python_code.invoke({"code": code})
    report = f"Generated Successfully\n\nExecution Result:\n\n{execution_result}"
    return {"report": report}

# ======================================
# GRAPH
# ======================================

builder = StateGraph(CrewState)
builder.add_node("developer", developer)
builder.add_node("tester", tester)
builder.add_edge(START, "developer")
builder.add_edge("developer", "tester")
builder.add_edge("tester", END)
langgraph_app = builder.compile()

# ======================================
# FASTAPI SETUP
# ======================================

class AgentInput(BaseModel):
    task: str = Field(description="Coding Task")

def format_input(x):
    task = x["task"] if isinstance(x, dict) else x.task
    return {"messages": [HumanMessage(content=task)]}

def format_output(state):
    return {"generated_code": state["code"], "report": state["report"]}

chain = (
    RunnableLambda(format_input)
    | langgraph_app
    | RunnableLambda(format_output)
).with_types(input_type=AgentInput)

app = FastAPI(title="LangGraph Coding Agent")

@app.get("/")
def health_check():
    return {"status": "LangGraph Coding Agent is running"}

add_routes(app, chain, path="/agent", playground_type="default")

# ======================================
# RUN SERVER
# ======================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    config = uvicorn.Config(app, host="0.0.0.0", port=port, loop="asyncio")
    server = uvicorn.Server(config)

    # Use the existing event loop to run the server as a background task
    loop = asyncio.get_event_loop()
    if not any(t.get_name() == 'uvicorn_server' for t in asyncio.all_tasks()):
        task = loop.create_task(server.serve(), name='uvicorn_server')
        print(f"Uvicorn server started on http://0.0.0.0:{port}")
