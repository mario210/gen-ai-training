"""
Exercise 06: Structured Output
This script demonstrates how to enforce structured JSON output from a language model
using Pydantic models and JsonOutputParser.
"""

import utils
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv(find_dotenv(usecwd=True))

MODEL_NAME = "openai/gpt-4o-mini"


class TranslationOutput(BaseModel):
    input_text_org: str = Field(description="Text of the input sentence.")
    input_text_translated: str = Field(
        description="Text of the input sentence translated."
    )
    translate_from_lang: str = Field(description="Translation from language.")
    translate_to_lang: str = Field(description="Translation to language.")


def structured_output_example():
    """Main function to setup and run a chain that produces structured JSON output."""
    model = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.1,
        base_url="https://openrouter.ai/api/v1",
        api_key=utils.get_api_key("OPENROUTER_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a translator assistant from one language to another. "
                "Make sure the output is valid JSON object with valid structure with fields input_text_org, input_text_translated, translate_from_lang, translate_to_lang.",
            ),
            (
                "user",
                "Translate this sentence {input} into {translate_to_lang}.",
            ),
        ]
    )

    chain = prompt | model | JsonOutputParser(pydantic_object=TranslationOutput)

    res = chain.invoke(
        {
            "input": "Tonight is the night.",
            "translate_to_lang": "Polish",
        }
    )
    print(res)


if __name__ == "__main__":
    structured_output_example()
