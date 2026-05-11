"""
Exercise 14:
"""

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
from pprint import pprint

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
    """
    response = requests.post(
        "http://localhost:8000/search",
        json={"query": query, "num_results": 5},
    )
    return str(response.json())


tools = [search_tool]

sys_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a personal chef. The user will give you a list of ingredients they have left over in their house.
Using the web search tool, search the web for recipes that can be made with the ingredients they have.
Return recipe suggestions and eventually the recipe instructions to the user, if requested.
""",
        ),
    ]
)

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.8,  # be creative
    base_url="https://openrouter.ai/api/v1",
    api_key=utils.get_api_key("OPENROUTER_API_KEY"),
)

# ---- AGENT ----
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a personal chef. The user will give you a list of ingredients they have left over in their house.
Using the web search tool, search the web for recipes that can be made with the ingredients they have.
Return recipe suggestions and eventually the recipe instructions to the user, if requested.
""",
)


if __name__ == "__main__":
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="I have some leftover chicken and rice. What can I make?"
                )
            ]
        },
    )

    pprint(response)
