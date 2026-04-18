"""
Exercise 13:
Demonstrates a basic Retrieval-Augmented Generation (RAG) pipeline using LangChain,
OpenAI embeddings/models (via OpenRouter), and a local Chroma vector database.
"""

import utils
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

MODEL_NAME = "openai/gpt-4o-mini"
EMBEDDED_LLM_MODEL_NAME = "text-embedding-3-small"


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

    # 2. Augmentation: Combine the retrieved documents' content to form the context
    prompt_context_info = ";".join(docs_page_content)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly assistant. You will shown the user request and the relevant information. Answer the request ONLY based on that information. Say 'I dont'know' if you don't know the information. Do not guess.",
            ),
            (
                "user",
                "Request {usr_query}. Context information: {rag_context_info}",
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
    llm_response = chain.invoke(
        {"usr_query": query, "rag_context_info": prompt_context_info}
    )

    print(llm_response)


if __name__ == "__main__":
    query = "How can I log to a file?"
    rag_example(query)
