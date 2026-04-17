"""
Exercise 08: Load PDF File

This script demonstrates how to load a PDF document using LangChain's PyMuPDF4LLMLoader.
It extracts the text content and metadata, prints the loaded documents to the console,
and saves them into a structured JSON file for further processing.
"""

import utils
from dotenv import load_dotenv, find_dotenv
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

load_dotenv(find_dotenv(usecwd=True))


def load_file_example():
    pdf_path = "../assets/howto-logging.pdf"

    # Load PDF using PyMuPDF4LLMLoader
    loader = PyMuPDF4LLMLoader(pdf_path)
    docs = loader.load()

    print(len(docs))
    # Print the whole loaded file
    print(docs)

    # Print content of first page
    print(docs[0].page_content)

    # Save loaded document as JSON file
    utils.save_langchain_docs_to_json(docs, "../assets/howto_logging.json")


if __name__ == "__main__":
    load_file_example()
