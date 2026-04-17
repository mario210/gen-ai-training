"""
Exercise 09:
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

import utils
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))


def chunking_example():
    docs = utils.load_langchain_docs_from_json("../assets/howto_logging.json")

    for i in range(len(docs)):
        processed_page_content = utils.clean_text(docs[i].page_content)
        docs[i].page_content = processed_page_content

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["#", "##", "###", "####", "\n\n", "\n", " ", ""],
    )

    docs_split = text_splitter.split_documents(docs)
    
    print(f"Total chunks created: {len(docs_split)}")

    # Check if chunking went well and if any chunks are oversized
    max_length = 0
    oversized_count = 0
    for i, doc in enumerate(docs_split):
        length = len(doc.page_content)
        max_length = max(max_length, length)
        if length > 100:
            oversized_count += 1
            
    print(f"Max chunk length: {max_length}")
    print(f"Oversized chunks (>100 chars): {oversized_count}")
    print("Chunking went well!" if oversized_count == 0 else "Warning: Oversized chunks detected.")


if __name__ == "__main__":
    chunking_example()
