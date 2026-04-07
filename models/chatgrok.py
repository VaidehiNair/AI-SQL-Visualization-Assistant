import os
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
groq_api_key = os.getenv("GROQ_API_KEY")


os.environ["GROQ_API_KEY"] = groq_api_key

llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")