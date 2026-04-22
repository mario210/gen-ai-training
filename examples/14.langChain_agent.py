"""
Exercise 14: Weather → Clothing Agent (LangChain)

This script demonstrates a simple LangChain agent that recommends clothing based on the weather.
"""

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

import requests
import utils
import warnings

# ---- SETUP ----
load_dotenv(find_dotenv(usecwd=True))
warnings.filterwarnings("ignore")

MODEL_NAME = "openai/gpt-4o-mini"


# ---- TOOLS ----
@tool
def search_tool(query: str) -> str:
    """
    Search the web using OrioSearch API.
    Use this to find current weather.
    """
    response = requests.post(
        "http://localhost:8000/search",
        json={"query": query, "num_results": 5},
    )
    return str(response.json())


@tool
def recommend_clothing(weather: str) -> str:
    """
    Recommend clothing based on weather description.
    """
    weather = weather.lower()

    if "snow" in weather or "freezing" in weather:
        return "Wear a heavy coat, gloves, scarf, and insulated boots."
    elif "rain" in weather or "shower" in weather:
        return "Bring a raincoat or umbrella and waterproof shoes."
    elif "hot" in weather or "30" in weather or "85" in weather:
        return "Wear light clothing like a t-shirt, shorts, and sunglasses."
    elif "cold" in weather or "10" in weather or "50" in weather:
        return "Wear a warm jacket or sweater."
    else:
        return "A light jacket or casual outfit should be fine."


tools = [search_tool, recommend_clothing]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Your job:
1. Check current weather using search_tool
2. Extract useful weather info
3. Call recommend_clothing
4. Return final recommendation

Always use tools when needed. Do not guess weather.
""",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=utils.get_api_key("OPENROUTER_API_KEY"),
)

# ---- AGENT ----
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a helpful assistant.
Steps:
1. Use search_tool to get current weather
2. Then use recommend_clothing
3. Return final answer
Do NOT guess weather.
""",
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "What should I wear today in Warsaw?"}
            ]
        }
    )

    print("\n=== Agent Execution Trace ===\n")
    for msg in result["messages"]:
        if msg.type == "human":
            print(f"User:\n{msg.content}\n")
        elif msg.type == "ai":
            # If AI is calling a tool
            if msg.tool_calls:
                print("AI (tool call):")
                for call in msg.tool_calls:
                    print(f"  → {call['name']}({call['args']})")
                print()
            else:
                print(f"AI (final answer):\n{msg.content}\n")
        elif msg.type == "tool":
            print(f"Tool [{msg.name}]:\n{msg.content}\n")
    print("=== End ===")
