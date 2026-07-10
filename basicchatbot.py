from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

def chatbot(state: State):
    return {
        "messages": [
            llm.invoke(state["messages"])
        ]
    }

graph_builder = StateGraph(State)

graph_builder.add_node("llmchatbot", chatbot)

graph_builder.add_edge(START, "llmchatbot")
graph_builder.add_edge("llmchatbot", END)

graph = graph_builder.compile()

# response = graph.invoke(
#     {
#         "messages": [
#             HumanMessage(content="Hello, how are you?")
#         ]
#     }
# )

# print(response["messages"][-1].content)
response = graph.invoke(
    {
        "messages": [
            HumanMessage(content="Calculate 145 × 12")
        ]
    }
)

print(response["messages"][-1].content)
png_data = graph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_data)

print("Graph saved as graph.png")


from langchain_tavily import TavilySearch, TavilyResearch
tool = TavilySearch(max_results=3)
response=tool.invoke("What is the capital of France?")
print(response)

##Custom function
def multiply(a: int, b: int) -> int:
    """Multiply two number a and b
    Args:
    a (int): The first number
    b (int): The second number
    Returns:
    int:output int
    """
tools=[tool, multiply]
llm_with_tools = llm.bind_tools(tools)
print (llm_with_tools)

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
def tool_calling_llm(state: State):
    return{"messages": [llm_with_tools.invoke(state["messages"])]}
builder = StateGraph(State)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)
builder.compile()

