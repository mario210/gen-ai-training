import os
import json
import re
import unicodedata
from pydantic import SecretStr
from langchain_core.documents import Document


def get_api_key(key: str):
    """Retrieves the key and securely wraps it."""
    raw_api_key = os.getenv(key)
    return SecretStr(raw_api_key) if raw_api_key else None


def encode_image(image_path: str) -> str:
    """Encodes an image to a base64 string."""
    import base64

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def save_langchain_docs_to_json(docs, file_path):
    docs_dict = [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(docs_dict, f, ensure_ascii=False, indent=4)


def load_langchain_docs_from_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        docs_dict = json.load(f)
    return [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in docs_dict
    ]


def clean_text(text: str) -> str:
    """
    Cleans text by normalizing unicode, removing URLs, emails, special characters,
    and extra whitespaces. Useful before data chunking.
    """
    if not text:
        return text

    # Normalize unicode characters (NFKC standardizes compatibility characters)
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove emails
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)

    # Remove special characters (keeps alphanumeric and whitespaces)
    # We replace them with space to avoid merging adjacent words
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove additional whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
