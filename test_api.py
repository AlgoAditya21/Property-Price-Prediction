import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the secret key from the .env file
load_dotenv()

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Send a test prompt
response = llm.invoke("Hi! Are you ready to act as a Bangalore Real Estate Advisor?")

print("AI Response:", response.content)
