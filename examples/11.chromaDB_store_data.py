"""
Exercise 11: Create and populate a Chroma vector database with documents.
"""

import utils
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

EMBEDDED_LLM_MODEL_NAME = "text-embedding-3-small"


def chromaDB_store_data_example():
    docs = utils.load_langchain_docs_from_json("../assets/howto_logging.json")
    print(f"Number of documents loaded: {len(docs)}")

    # Context Enrichment: Inject metadata directly into the page_content.
    # This ensures both the embedding model and the LLM have explicit knowledge 
    # of where this chunk of text came from.
    for doc in docs:
        title = doc.metadata.get("title", "Unknown Title")
        author = doc.metadata.get("author", "Unknown Author")
        page = doc.metadata.get("page", "Unknown Page")
        
        doc.page_content = f"Document Title: {title}\nAuthor: {author}\nPage Number: {page}\n\nContent:\n{doc.page_content}"

    embedding_model = OpenAIEmbeddings(
        model=EMBEDDED_LLM_MODEL_NAME,
        base_url="https://openrouter.ai/api/v1",
        api_key=utils.get_api_key("OPENROUTER_API_KEY"),
    )

    # create vector store
    Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory="../assets/chroma_db",
    )


if __name__ == "__main__":
    chromaDB_store_data_example()
