"""
Exercise 10: Generating Embeddings from a File
This script demonstrates how to load documents from a JSON file and generate embeddings
for their contents using the OpenAI Embeddings API via OpenRouter.
"""

import utils
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

EMBEDDED_LLM_MODEL_NAME = "text-embedding-3-small"


def embedding_example():
    docs = utils.load_langchain_docs_from_json("../assets/howto_logging.json")
    print(f"Number of documents loaded: {len(docs)}")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDED_LLM_MODEL_NAME,
        base_url="https://openrouter.ai/api/v1",
        api_key=utils.get_api_key("OPENROUTER_API_KEY"),
    )

    docs_content = [doc.page_content for doc in docs]

    # Generate embeddings for the extracted text contents
    docs_embeded = embeddings.embed_documents(texts=docs_content)
    print(f"Number of embeddings generated: {len(docs_embeded)}")

    # Print the dimensionality of the first embedding (number of features/floats in the vector)
    print(f"Dimensionality of the embedding: {len(docs_embeded[0])}")


if __name__ == "__main__":
    embedding_example()
