"""
Exercise 12: Search for a query in an existing Chroma vector database.
"""

import utils
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

EMBEDDED_LLM_MODEL_NAME = "text-embedding-3-small"


def chromaDB_search_query_example():
    embedding_model = OpenAIEmbeddings(
        model=EMBEDDED_LLM_MODEL_NAME,
        base_url="https://openrouter.ai/api/v1",
        api_key=utils.get_api_key("OPENROUTER_API_KEY"),
    )

    query = "How can I log to a file?"

    # search in vector database
    vector_store = Chroma(
        embedding_function=embedding_model, persist_directory="../assets/chroma_db"
    )

    # version 1 : search by similarity
    results = vector_store.similarity_search(query)
    print(f"[similarity] Search results ({len(results)}): {results}")

    # version 2 : search by relevance
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    results_retriever = retriever.invoke(query)
    print(f"[retriever] Search results ({len(results_retriever)}): {results_retriever}")


if __name__ == "__main__":
    chromaDB_search_query_example()
