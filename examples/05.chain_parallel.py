"""
Exercise 05: Parallel Chains
This script demonstrates how to run multiple chains in parallel using RunnableParallel
and RunnableLambda.
"""

import utils
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

MODEL_NAME = "openai/gpt-4o-mini"


def chain_parallel_example():
    """Main function to setup and run parallel LangChain chains."""
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

    story_chain = prompt | model | StrOutputParser()

    witcher_variables = {"character": "Geralt of Rivia", "location": "Novigrad"}
    cyberpunk_variables = {"character": "V", "location": "Night city"}

    map_chain = RunnableParallel(
        story_1=RunnableLambda(lambda x: story_chain.invoke(witcher_variables)),
        story_2=RunnableLambda(lambda x: story_chain.invoke(cyberpunk_variables)),
    )

    result = map_chain.invoke({})

    print("Wither story:")
    print(result["story_1"])
    print("Cyberpunk story:")
    print(result["story_2"])


if __name__ == "__main__":
    chain_parallel_example()
