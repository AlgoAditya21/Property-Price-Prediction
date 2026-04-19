import hashlib
import os
from typing import List

from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()


class LocalHashEmbeddings(Embeddings):
	"""Dependency-free embeddings used when cloud embeddings are unavailable."""

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


def _get_embedding_for_existing_db(persist_dir: str):
	"""Pick an embedding function compatible with the already built Chroma DB."""
	embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")

	candidates = [
		(
			"Google",
			lambda: GoogleGenerativeAIEmbeddings(model=embedding_model),
			f"Using Google embeddings for retrieval: {embedding_model}",
		),
		(
			"FastEmbed",
			lambda: FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
			"Using FastEmbed fallback embeddings for retrieval.",
		),
		(
			"LocalHash",
			lambda: LocalHashEmbeddings(),
			"Using local hash embeddings for retrieval.",
		),
	]

	embedding_fn = None
	for name, build_embedding, success_message in candidates:
		try:
			embedding_fn = build_embedding()
			probe_db = Chroma(persist_directory=persist_dir, embedding_function=embedding_fn)
			probe_db.similarity_search("whitefield", k=1)
			print(success_message)
			return embedding_fn
		except Exception as exc:
			print(f"{name} embedding retriever unavailable ({exc}).")
			embedding_fn = None

	if embedding_fn is None:
		print("All embedding backends failed. Using LocalHash embeddings as fallback.")
		return LocalHashEmbeddings()

	raise RuntimeError("No compatible embedding backend found for the current Chroma DB.")


def _get_chat_llm() -> ChatGoogleGenerativeAI:
	"""Select the first available chat model for the current API endpoint."""
	candidate_models = [
		os.getenv("GOOGLE_CHAT_MODEL", "models/gemini-2.0-flash"),
		"models/gemini-2.5-flash",
		"models/gemini-flash-latest",
		"models/gemini-pro-latest",
		"gemini-2.0-flash",
	]

	tried = []
	for model_name in candidate_models:
		if model_name in tried:
			continue
		tried.append(model_name)
		try:
			llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
			llm.invoke("ping")
			print(f"Using chat model: {model_name}")
			return llm
		except Exception as exc:
			print(f"Chat model unavailable ({model_name}): {exc}")

	raise RuntimeError(
		"No supported Gemini chat model was found. Set GOOGLE_CHAT_MODEL to a model available on your API key."
	)


class RealEstateRAGAgent:
	def __init__(self, llm: ChatGoogleGenerativeAI, retriever, prompt: ChatPromptTemplate):
		self.llm = llm
		self.retriever = retriever
		self.prompt = prompt

	def _build_context(self, docs: List[Document]) -> str:
		if not docs:
			return "No relevant report context found."
		return "\n\n".join(doc.page_content for doc in docs)

	def invoke(self, inputs: dict) -> dict:
		user_input = inputs.get("input", "")
		docs = self.retriever.invoke(user_input)
		context = self._build_context(docs)
		messages = self.prompt.format_messages(input=user_input, context=context)
		response = self.llm.invoke(messages)
		return {"answer": response.content}


def get_real_estate_agent():
	llm = _get_chat_llm()
	persist_dir = "./chroma_db"
	embeddings = _get_embedding_for_existing_db(persist_dir)
	vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
	retriever = vector_db.as_retriever(search_kwargs={"k": 3})
	system_prompt = (
		"You are an expert, professional Bangalore Real Estate Advisor. "
		"Use only the retrieved context from market reports to answer the user precisely. "
		"If context is missing for a claim, say you do not have exact report data and provide cautious general advice. "
		"If a 'Predicted Price' is provided, incorporate it into tailored investment guidance for the mentioned locality.\n\n"
		"Context from Market Reports:\n{context}"
	)

	prompt = ChatPromptTemplate.from_messages(
		[
			("system", system_prompt),
			("human", "{input}"),
		]
	)

	return RealEstateRAGAgent(llm=llm, retriever=retriever, prompt=prompt)


if __name__ == "__main__":
	print("Initializing Agent...")
	agent = get_real_estate_agent()

	test_price = "INR 1.2 Cr"
	test_query = (
		f"My ML model just predicted a price of {test_price} for a 3BHK in Whitefield. "
		"Based on current market trends, is this a good investment?"
	)

	print(f"\nUser: {test_query}\n")
	print("Agent is searching the database and thinking...\n")

	response = agent.invoke({"input": test_query})
	print(f"Agent:\n{response['answer']}")
