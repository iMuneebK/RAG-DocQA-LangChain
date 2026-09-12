import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
import config

class RAGEngine:
    def __init__(self):
        # Initialize embeddings
        if "sentence-transformers" in config.EMBEDDING_MODEL:
            self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        else:
            self.embeddings = OpenAIEmbeddings()

        self.vector_store = None
        self.chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def build_vector_store(self, chunks):
        """Creates a FAISS vector store from document chunks."""
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(config.VECTOR_STORE_PATH)
        self._initialize_chain()

    def load_vector_store(self):
        """Loads an existing FAISS vector store."""
        if os.path.exists(config.VECTOR_STORE_PATH):
            self.vector_store = FAISS.load_local(config.VECTOR_STORE_PATH, self.embeddings, allow_dangerous_deserialization=True)
            self._initialize_chain()
            return True
        return False

    def _initialize_chain(self):
        """Initializes the conversational retrieval chain."""
        llm = ChatOpenAI(model_name=config.LLM_MODEL, temperature=0.2)
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory
        )

    def ask(self, query):
        """Asks a question to the RAG pipeline."""
        if not self.chain:
            raise ValueError("Vector store and chain are not initialized.")
        response = self.chain.invoke({"question": query})
        return response['answer']
