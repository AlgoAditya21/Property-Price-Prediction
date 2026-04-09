import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load the secret key
load_dotenv()

def create_vector_db():
    # 1. Load PDFs from the 'data' folder
    print("Loading PDFs from the data folder...")
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()
    
    if not docs:
        print("Error: No PDFs found in the 'data' folder! Please add some.")
        return

    print(f"Loaded {len(docs)} pages.")

    # 2. Split the text into manageable chunks
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # 3. Convert to embeddings and save to ChromaDB
    print("Creating vector database... (This might take a few seconds)")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # This creates a folder called 'chroma_db' locally
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )

    print("✅ Success! RAG database created and saved in the './chroma_db' folder.")

if __name__ == "__main__":
    create_vector_db()
