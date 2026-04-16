"""
Exercise 07: Semantic Router Chain
This script demonstrates routing user queries to different LangChain chains based on embedding similarity.
"""

import utils
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utils.math import cosine_similarity
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

LLM_MODEL_NAME = "openai/gpt-4o-mini"
EMBEDDED_LLM_MODEL_NAME = "text-embedding-3-small"


def router_chain_example():
    """Main function to setup and run the semantic router chain."""
    # 1. Initialize the Language Model (LLM)
    model = ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=0.2,
        base_url="https://openrouter.ai/api/v1",
        api_key=utils.get_api_key("OPENROUTER_API_KEY"),
    )

    # 2. Define prompt templates and chains for specific tasks (Order and FAQ)
    order_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a cafe ordering assistant. Acknowledge the user's order enthusiastically.",
            ),
            ("user", "{query}"),
        ]
    )
    order_chain = order_prompt | model | StrOutputParser()

    faq_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a cafe FAQ assistant. Answer questions based on the rule that opening hours are 8:00 AM to 6:00 PM.",
            ),
            ("user", "{query}"),
        ]
    )
    faq_chain = faq_prompt | model | StrOutputParser()

    # 3. Initialize the Embeddings model for semantic routing
    embeddings = OpenAIEmbeddings(
        model=EMBEDDED_LLM_MODEL_NAME,
        base_url="https://openrouter.ai/api/v1",
        api_key=utils.get_api_key("OPENROUTER_API_KEY"),
    )

    # 4. Define Semantic Router logic using Embeddings
    def route_query(info: dict):
        """
        Routes the user query to the appropriate chain (order or FAQ)
        based on semantic similarity using cosine similarity scores.
        """
        query = info["query"]

        # Representative text for our target intents
        order_examples = [
            "I want to order a coffee",
            "Can I get a cappuccino?",
            "I'd like some food",
            "Give me a latte please",
        ]

        faq_examples = [
            "What are your opening hours?",
            "When do you open?",
            "Are you open now?",
            "What time do you close?",
        ]

        # Generate vectors
        order_emb = np.mean([embeddings.embed_query(x) for x in order_examples], axis=0)
        faq_emb = np.mean([embeddings.embed_query(x) for x in faq_examples], axis=0)
        query_emb = embeddings.embed_query(query)

        # Calculate similarity matrix and find the index with the highest score
        similarities = cosine_similarity([query_emb], [order_emb, faq_emb])
        most_similar_index = similarities[0].argmax()

        if most_similar_index == 0:
            print(f"--> [Semantic Router] Triggering ORDER LLM for: '{query}'")
            return order_chain
        else:
            print(f"--> [Semantic Router] Triggering FAQ LLM for: '{query}'")
            return faq_chain

    # 5. Combine everything into a dynamic router chain
    # RunnableParallel resolves the type hints while formatting the raw string
    router_chain = RunnableParallel(query=RunnablePassthrough()) | RunnableLambda(
        route_query
    )

    # 6. Test out the specified queries
    print("\n--- Test Case 1 ---")
    print(router_chain.invoke("I would like to order cappucino."))

    print("\n--- Test Case 2 ---")
    print(router_chain.invoke("What are opening hours?"))


if __name__ == "__main__":
    router_chain_example()
