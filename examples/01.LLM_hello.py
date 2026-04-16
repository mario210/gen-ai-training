"""
Exercise 01: Introduction to Large Language Models (LLMs) with LangChain and OpenRouter.
This file demonstrates how to connect to an external LLM API (OpenRouter) using LangChain's OpenAI wrapper.
It covers loading environment variables, setting up a ChatPromptTemplate, and executing a basic LCEL (LangChain Expression Language) chain.
"""

import utils
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv(usecwd=True))

MODEL_NAME = "openai/gpt-4o-mini"


def print_hi():
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.5,
        base_url="https://openrouter.ai/api/v1",
        api_key=utils.get_api_key("OPENROUTER_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a friendly assistant."),
            ("user", "Hello, my name is {name}"),
        ]
    )

    chain = prompt | llm

    res = chain.invoke({"name": "Ala"})
    print(res.content)


if __name__ == "__main__":
    print_hi()
