import os
import shutil
import hashlib
from typing import List
from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


class LocalHashEmbeddings:
    """Dependency-free fallback embeddings based on token hashing."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = text.lower().split()

        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign

        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

def create_vector_db():
    print("Loading PDFs from the data folder...")
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()
    
    if not docs:
        print("Error: No PDFs found in the 'data' folder! Please add some.")
        return

    print(f"Loaded {len(docs)} pages.")

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Creating vector database... (This might take a few seconds)")
    embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
    persist_dir = "./chroma_db"

    shutil.rmtree(persist_dir, ignore_errors=True)

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
        print("Indexed with Google embeddings.")
    except Exception as exc:
        print(f"Google embeddings unavailable ({exc}). Trying FastEmbed fallback...")
        try:
            fast_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
            Chroma.from_documents(
                documents=chunks,
                embedding=fast_embeddings,
                persist_directory=persist_dir,
            )
            print("Indexed with FastEmbed fallback embeddings.")
        except Exception as fast_exc:
            print(
                f"FastEmbed fallback unavailable ({fast_exc}). "
                "Falling back to local hash embeddings..."
            )
            local_embeddings = LocalHashEmbeddings()
            Chroma.from_documents(
                documents=chunks,
                embedding=local_embeddings,
                persist_directory=persist_dir,
            )
            print("Indexed with local hash fallback embeddings.")

    print("✅ Success! RAG database created and saved in the './chroma_db' folder.")

if __name__ == "__main__":
    create_vector_db()
