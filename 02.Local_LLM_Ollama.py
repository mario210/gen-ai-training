"""
Exercise 02: Running Local LLMs with Ollama.
This file demonstrates how to run and interact with a local language model using Ollama and LangChain.
It shows that you can query a locally hosted model (e.g., qwen3:4b) directly without relying on external cloud APIs.
"""

from langchain_ollama import ChatOllama


def run_local_llm():
    llm = ChatOllama(model="qwen3:4b", temperature=0.2)

    res = llm.invoke("What is Ollama. Response in one sentence.")
    print(res.content)


if __name__ == "__main__":
    run_local_llm()
