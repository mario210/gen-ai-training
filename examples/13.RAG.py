"""
Exercise 13: RAG with Prompt Caching
Demonstrates a basic Retrieval-Augmented Generation (RAG) pipeline using LangChain,
OpenAI embeddings/models (via OpenRouter), a local Chroma vector database,
and in-memory prompt caching to reduce latency and API costs.
"""

import time
import utils
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

# Enable in-memory caching for LLM responses
set_llm_cache(InMemoryCache())

MODEL_NAME = "openai/gpt-4o-mini"
EMBEDDED_LLM_MODEL_NAME = "text-embedding-3-small"


def invoke_and_check_cache(chain, input_data: dict) -> str:
    """Invokes the chain and determines if the response was cached based on execution time."""
    start_time = time.time()
    response = chain.invoke(input_data)
    elapsed_time = time.time() - start_time
    
    # API calls typically take >0.5s. In-memory cache hits are near instantaneous (<0.1s).
    is_cached = elapsed_time < 0.1
    print(f"\n[Info] LLM Generation took {elapsed_time:.4f} seconds (Cache Hit: {is_cached})")
    return response


def rag_example(query: str):
    # Initialize the embedding model used to encode the user's query
    embedding_model = OpenAIEmbeddings(
        model=EMBEDDED_LLM_MODEL_NAME,
        base_url="https://openrouter.ai/api/v1",
        api_key=utils.get_api_key("OPENROUTER_API_KEY"),
    )

    # Load the existing Chroma vector store from the local directory
    vector_store = Chroma(
        embedding_function=embedding_model, persist_directory="../assets/chroma_db"
    )
    # 1. Retrieval: Fetch top 5 most similar documents from the vector store
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    docs = retriever.invoke(query)
    docs_page_content = [doc.page_content for doc in docs]

    # 2. Augmentation: Define the prompt template
    prompt_context_info = ";".join(docs_page_content)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly assistant. You will be shown the user request and the relevant information. Answer the request ONLY based on that information. Say 'I don't know' if you don't know the information. Do not guess.",
            ),
            (
                "user",
                "Request: {usr_query}\n\nContext information:\n{rag_context_info}",
            ),
        ]
    )

    # 3. Generation: Initialize the LLM and generate an answer using the augmented prompt
    model = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.1,
        base_url="https://openrouter.ai/api/v1",
        api_key=utils.get_api_key("OPENROUTER_API_KEY"),
    )

    # Build and invoke the LangChain Expression Language (LCEL) chain
    chain = prompt_template | model | StrOutputParser()
    
    llm_response = invoke_and_check_cache(
        chain,
        {"usr_query": query, "rag_context_info": prompt_context_info}
    )

    print("Response:\n" + llm_response)


if __name__ == "__main__":
    query = "How can I log to a file?"
    print("--- First Run (Expected Cache Miss) ---")
    rag_example(query)
    print("\n--- Second Run (Expected Cache Hit) ---")
    rag_example(query)
