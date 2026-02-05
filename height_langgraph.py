# height_langgraph.py

import os
from dotenv import load_dotenv
load_dotenv()

from typing_extensions import TypedDict
from typing import Annotated

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

from Height_Weight_Predict import predict

# ---------------- ENV ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

# ---------------- PROMPT ----------------
template = """
You are a health assistant.

STRICT RULES:
1. Height is given in cm.
2. Use BMI range 18.5–24.9.
3. You MUST compute a weight RANGE.
4. Output ONLY in this format:

Healthy weight range for this height is: X kg – Y kg
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{Height}")
])

# ---------------- TOOL ----------------
@tool
def weight_predict(height: float) -> float:
    """Predicts weight from height using ML model"""
    value= float(predict(height))
    return round(value,2)
# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    temperature=0,
).bind_tools([weight_predict])

chain = prompt | llm

# ---------------- STATE ----------------
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# ---------------- NODES ----------------
def llm_node(state: State):
    last = state["messages"][-1]

    if isinstance(last, HumanMessage):
        return {
            "messages": [chain.invoke({"Height": last.content})]
        }

    return {"messages": []}

# ---------------- GRAPH ----------------
graph = StateGraph(State)

graph.add_node("llm", llm_node)
graph.add_node("tools", ToolNode([weight_predict]))

graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", tools_condition)
graph.add_edge("tools", "llm")
graph.add_edge("llm", END)

height_weight_app = graph.compile()
