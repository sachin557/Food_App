import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.prompts import ChatPromptTemplate
from Height_Weight_Predict import predict
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
output=StrOutputParser()

template = """
You are a health assistant.

You will receive:
- a user's height (in cm)
- a predicted weight from a tool

Your task:
1. Check if the predicted weight is reasonable for the height
2. Use BMI range 18.5–24.9 as a guideline
3. If the tool result is reasonable, keep it
4. If not, calculate and provide a corrected healthy weight
5. Always explain briefly what you did
"""

prompt=ChatPromptTemplate.from_messages(
    [("system",template),("human","{Height}")]
)
@tool
def weight_predict(height:float) -> float:
    """ will predict the weight based on the height provided using linear model """
    return predict(height)
llm=ChatGroq(model="llama-3.1-8b-instant",groq_api_key=GROQ_API_KEY).bind_tools([weight_predict])
chain= prompt|llm
class State(TypedDict):
    messages: Annotated[list[AnyMessage],add_messages]
def llm_node(state:State):
    last_msg=state["messages"][-1]
    if isinstance(last_msg,HumanMessage):
        height=state["messages"][-1].content

        return {
            "messages":[chain.invoke({"Height":height})]
        }
    return {
        "messages":[llm.invoke(state["messages"])]
    }    
### graph ###
graph=StateGraph(State)
graph.add_node("llm",llm_node)
graph.add_node("tools",ToolNode([weight_predict]))
graph.add_edge(START,"llm")
graph.add_conditional_edges("llm",tools_condition)
graph.add_edge("tools","llm")
app=graph.compile()
input_height=input("enter height in cm")
result=app.invoke(
    {
    "messages":[HumanMessage(content=input_height)]
    }
)
for msg in result["messages"]:
    print(msg.content)