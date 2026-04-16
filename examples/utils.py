import os
from pydantic import SecretStr


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
