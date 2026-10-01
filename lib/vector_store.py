"""Vector store and retrieval helpers."""

from lib.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
)


def build_embeddings():
    """Build the local Ollama embeddings object used by Chroma."""

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def build_vector_store():
    """Build the Chroma vector store used by the RAG pipeline."""

    from langchain_chroma import Chroma

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=build_embeddings(),
    )


def retrieve_context(question, *, vector_store=None, top_k=DEFAULT_TOP_K):
    """Retrieve scored documents for a question."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-blank string.")

    if vector_store is None:
        vector_store = build_vector_store()

    return vector_store.similarity_search_with_score(question, k=top_k)
