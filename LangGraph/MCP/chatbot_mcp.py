from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()  # Load environment variables from .env file

llm = ChatOpenAI(model="gpt-5")

# MCP client for local FastMCP server
client = MultiServerMCPClient(
    {
        "expense": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)


# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():

    tools = await client.get_tools()

    print(tools)

    llm_with_tools = llm.bind_tools(tools)

    # nodes
    async def chat_node(state: ChatState):

        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(tools)

    # defining graph and nodes
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # defining graph connections
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()

    return chatbot

async def main():

    chatbot = await build_graph()

    # running the graph
    add_expenses = await chatbot.ainvoke({"messages": [HumanMessage(content="add 1000 to my expenses on 1st of Dec 2025, 200 on 2nd Dec 2025, 90 on 3rd Dec 2025, 100 on 4th Dec 2025, 250 on 5th Dec 2025, 300 on 6th Dec 2025, 50 on 7th Dec 2025 and 180 on 8th Dec 2025 all the expense are of food category")]})
    print(add_expenses['messages'][-1].content)
    print("--------------------------------")
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="Give me all my expenses for the month of Dec from 1 Dec to 31 Dec")]})
    print(result['messages'][-1].content)
    print("--------------------------------")

if __name__ == '__main__':
    asyncio.run(main())