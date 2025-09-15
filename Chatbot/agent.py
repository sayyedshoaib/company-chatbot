from google.adk.agents import LlmAgent
import os
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

root_agent = LlmAgent(
    name="chatbot",
    model="gemini-2.0-flash",
    description="You are chatbot to answer user query",
    instruction = '''
        You are chatbot to answer user query
    ''',)