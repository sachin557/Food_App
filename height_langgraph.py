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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

template = """
You are a health assistant.

STRICT RULES:
1. You receive height in cm.
2. Compute BMI healthy range using BMI 18.5–24.9.
3. DO NOT return a single value.
4. Return EXACTLY in this format:

Healthy weight range for this height is: X kg – Y kg
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{Height}")
])

@tool
def weight_predict(height: float) -> float:
    """Predicts weight from height using ML model"""
    return predict(height)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    temperature=0,
).bind_tools([weight_predict])

chain = prompt | llm

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def llm_node(state: State):
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        return {
            "messages": [chain.invoke({"Height": last.content})]
        }
    return {"messages": []}

graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tools", ToolNode([weight_predict]))
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", tools_condition)
graph.add_edge("tools", "llm")

height_weight_app = graph.compile()
