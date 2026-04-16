"""
Exercise 04: Basic LangChain Chain
This script demonstrates how to create a basic LangChain chain using a prompt template,
a chat model, and a string output parser.
"""

import utils
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv(usecwd=True))

MODEL_NAME = "openai/gpt-4o-mini"


def chain_example():
    """Main function to setup and run a basic LangChain chain."""
    model = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.7,  # let's give the model creativity
        base_url="https://openrouter.ai/api/v1",
        api_key=utils.get_api_key("OPENROUTER_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a master storyteller. Create a short, engaging story based on the user's request.",
            ),
            (
                "user",
                "Tell me a story about a character named {character} who is in {location}.",
            ),
        ]
    )

    chain = prompt | model | StrOutputParser()

    res = chain.invoke(
        {
            "character": "Frodo Baggins",
            "location": "Mordor",
        }
    )
    print(res)


if __name__ == "__main__":
    chain_example()
