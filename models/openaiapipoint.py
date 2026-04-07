import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

_ = load_dotenv(find_dotenv())
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    model="gpt-oss-20b",
    temperature=0.7
)

