from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import config

if TYPE_CHECKING:
    from vectordb.store import VectorStore

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful assistant with access to a continuously
updated knowledge base. Use the retrieved context below to answer the
question accurately. If the context does not contain enough information,
say so honestly — do not fabricate facts.

Context:
{context}"""

class RAGChatbot:
    def __init__(self, store: "VectorStore") -> None:
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        self._llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.2,
            openai_api_key=config.OPENAI_API_KEY,
        )
        self._retriever = store.as_retriever(k=5)
        self._chat_history = []
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        logger.info("RAG chatbot ready (model: %s).", config.LLM_MODEL)

    def chat(self, question: str) -> dict:
        docs = self._retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        messages = self._prompt.format_messages(
            context=context,
            chat_history=self._chat_history,
            input=question,
        )
        response = self._llm.invoke(messages)
        answer = response.content
        self._chat_history.append(HumanMessage(content=question))
        self._chat_history.append(AIMessage(content=answer))
        self._chat_history = self._chat_history[-10:]
        sources = list({doc.metadata.get("source", "unknown") for doc in docs})
        return {"answer": answer, "sources": sources}

    def clear_history(self) -> None:
        self._chat_history = []
