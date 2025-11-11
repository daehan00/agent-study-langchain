from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv(".env")
llm_large = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=1.0
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0
)

llm_small = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=1.0
)