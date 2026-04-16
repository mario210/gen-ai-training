"""
Exercise 03: Introduction to Large Multimodal Models (LMMs) with Vision capabilities.
This file demonstrates how to use a multimodal model (e.g., Gemini via OpenRouter) to analyze an image.
It covers encoding a local image to base64 and constructing a structured HumanMessage containing both text and image data.
"""

import utils
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv(find_dotenv(usecwd=True))

# For multimodal tasks, it's best to use a model that explicitly supports vision.
# "gemini-2.5-flash-image" is a good choice available through OpenRouter.
MODEL_NAME = "google/gemini-2.5-flash-image"


def describe_image(image_path: str, prompt_text: str):
    """
    Uses a multimodal model to describe an image.

    Args:
        image_path: The path to the image file.
        prompt_text: The text prompt to send with the image.
    """
    # Instantiate the ChatOpenAI model, configured for OpenRouter
    model = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.5,
        base_url="https://openrouter.ai/api/v1",
        api_key=utils.get_api_key("OPENROUTER_API_KEY"),
    )

    # Encode the image to a base64 string using the utility function
    base64_image = utils.encode_image(image_path)

    # Create a message for the model, including the text prompt and the image
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
    )

    # Invoke the model with the message and get the response
    response = model.invoke([message])

    print(response.content)


if __name__ == "__main__":
    image_file = "../assets/istockphoto-683494078-1024x1024.jpg"
    question = "What is in this image?"
    try:
        describe_image(image_path=image_file, prompt_text=question)
    except FileNotFoundError:
        print(f"Error: The image file was not found at '{image_file}'.")
        print(
            "Please update the 'image_file' variable with a correct path to your image."
        )
