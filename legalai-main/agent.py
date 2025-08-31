import os as OS
import utils as Utils

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


class NewsChat:
    store = {}
    session_id = ""
    rag_chain = None

    def __init__(self, article_id: str):
        """
        Initialize the chat agent using Google Gemini for both LLM and embeddings.

        Requirements:
        - Environment variable GOOGLE_API_KEY must be set.
        - Package langchain-google-genai must be installed.
        - The Chroma collection should use the same embedding model ("models/text-embedding-004")
          that was used at indexing time for best results.
        """
        google_api_key = OS.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Set it before creating NewsChat."
            )

        # Embeddings: Google Generative AI (Gemini) text embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=google_api_key,
            model="models/text-embedding-004",
        )

        # Chat LLM: Google Gemini
        llm = ChatGoogleGenerativeAI(
            google_api_key=google_api_key,
            model="gemini-2.5-flash",
            temperature=0.2,
        )

        self.session_id = article_id

        # Vector store (persisted) with matching embeddings
        db = Chroma(
            persist_directory=Utils.DB_FOLDER,
            embedding_function=embeddings,
            collection_name="collection_1",
        )
        retriever = db.as_retriever()

        # Prompts
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference "
            "context in the chat history, formulate a standalone question which can be "
            "understood without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        self.rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def ask(self, question: str) -> str:
        response = self.rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": self.session_id}},
        )["answer"]
        return response