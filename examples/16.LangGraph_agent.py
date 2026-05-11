"""
Exercise 15: Weather → Clothing Agent (LangGraph)

This script demonstrates a simple LangGraph agent that recommends clothing based on the weather.
"""

from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv, find_dotenv
import requests
import utils
import json
import warnings

# ---- SETUP ----
load_dotenv(find_dotenv(usecwd=True))
warnings.filterwarnings("ignore")

MODEL_NAME = "openai/gpt-4o-mini"


# ---- STATE ----
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ---- TOOLS ----
@tool
def search_tool(query: str) -> str:
    """Search the web using OrioSearch API."""
    response = requests.post(
        "http://localhost:8000/search",
        json={"query": query, "num_results": 5},
    )
    return str(response.json())


@tool
def recommend_clothing(weather: str) -> str:
    """Recommend clothing based on weather."""
    weather = weather.lower()

    if "snow" in weather or "freezing" in weather:
        return "Wear a heavy coat, gloves, scarf, and boots."
    elif "rain" in weather:
        return "Bring a raincoat and waterproof shoes."
    elif "hot" in weather or "30" in weather:
        return "Wear light clothes like a t-shirt and shorts."
    elif "cold" in weather or "10" in weather:
        return "Wear a warm jacket."
    else:
        return "A light jacket should be fine."


@tool
def news_summarizer_tool(news_content: str) -> str:
    """
    Summarize news articles from search results.

    :param news_content: Raw news content or search results, expected to be a JSON string of a list of articles.
    :return: A formatted summary of the news
    """
    try:
        articles = json.loads(news_content)
    except json.JSONDecodeError:
        # If it's not a JSON list, try just returning it back for LLM to summarize, or basic formatting
        return f"Summary of content:\n{news_content}"

    if not isinstance(articles, list):
        # Could be a dict if it's a single result wrapper
        if isinstance(articles, dict) and "results" in articles:
            articles = articles["results"]
        else:
            return f"Summary of content:\n{news_content}"

    summary_parts = []
    for i, article in enumerate(articles, 1):
        if not isinstance(article, dict):
            continue
        headline = article.get("title", "No headline available")
        snippet = article.get("snippet", article.get("content", "No summary available"))
        date = article.get("date", article.get("published_date", "No date available"))

        summary_parts.append(
            f"Article {i}:\n"
            f"Headline: {headline}\n"
            f"Date: {date}\n"
            f"Main Points:\n- {snippet}\n"
        )

    if not summary_parts:
        return "No articles found to summarize."

    return "\n".join(summary_parts)


tools = [search_tool, recommend_clothing, news_summarizer_tool]
tools_by_name = {tool.name: tool for tool in tools}


# ---- MODEL ----
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant that thinks step-by-step and uses tools when needed.

When responding to queries:
1. First, think about what information you need
2. Use available tools if you need current data or specific capabilities  
3. Provide clear, helpful responses based on your reasoning and any tool results

Always explain your thinking process to help users understand your approach.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=utils.get_api_key("OPENROUTER_API_KEY"),
)

model = prompt | llm.bind_tools(tools)


# ---- NODES ----
def call_model(state: AgentState):
    response = model.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def tool_node(state: AgentState):
    outputs = []
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])

        outputs.append(
            ToolMessage(
                content=json.dumps(result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": outputs}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return "end"

    return "continue"


def print_stream(result: dict = None):
    global msg, call
    for msg in result["messages"]:
        if msg.type == "human":
            print(f"👤 User:\n{msg.content}\n")
        elif msg.type == "ai":
            if msg.tool_calls:
                print("🤖 AI (tool call):")
                for call in msg.tool_calls:
                    print(f"  → {call['name']}({call['args']})")
                print()
            else:
                print(f"🤖 AI (final answer):\n{msg.content}\n")
        elif msg.type == "tool":
            print(f"🛠 Tool [{msg.name}]:\n{msg.content}\n")


# ---- GRAPH ----

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

workflow.add_edge("tools", "agent")

app = workflow.compile()


# ---- MAIN ----

if __name__ == "__main__":

    # Test 1: Weather and Clothing
    print("Running Test 1: Weather and Clothing...")
    result_1 = app.invoke(
        {"messages": [HumanMessage(content="What should I wear today in Warsaw?")]}
    )

    print("\n=== Agent Execution Trace (Test 1) ===\n")
    print_stream(result_1)
    print("=== End Test 1 ===\n")

    # Test 2: AI News Summarization
    print("Running Test 2: AI News Summarization...")
    result_2 = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Find recent AI news and summarize the top 3 articles"
                )
            ]
        }
    )

    print("\n=== Agent Execution Trace (Test 2) ===\n")
    print_stream(result_2)
    print("=== End Test 2 ===")
