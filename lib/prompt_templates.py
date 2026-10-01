"""Prompt template helpers for the LangChain RAG workflow."""

SYSTEM_PROMPT = """
You are an internal reliability assistant.

Answer using only the approved retrieved context provided by the backend.
If the context does not contain enough information to answer reliably, say that
you do not have enough approved context to answer. Do not invent policies,
procedures, metrics, or incident steps.

Keep the response concise, specific, and useful to an engineer during an incident.
"""

HUMAN_PROMPT = """
Approved retrieved context:
{context}

Question:
{question}
"""


def build_rag_prompt():
    """Build the reusable LangChain prompt template for RAG answers."""

    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT.strip()),
            ("human", HUMAN_PROMPT.strip()),
        ]
    )
