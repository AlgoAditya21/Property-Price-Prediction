import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

response = llm.invoke("Hi! Are you ready to act as a Bangalore Real Estate Advisor?")

print("AI Response:", response.content)
