import streamlit as st
import os
from document_loader import DocumentLoader
from rag_engine import RAGEngine

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")

st.title("📚 RAG Document Q&A System")
st.markdown("Upload documents (PDF, TXT, DOCX) and ask questions about them!")

@st.cache_resource
def get_engine():
    return RAGEngine()

engine = get_engine()
doc_loader = DocumentLoader()

# Sidebar for file upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload your documents here", 
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        with st.spinner("Processing..."):
            all_chunks = []
            for file in uploaded_files:
                chunks = doc_loader.process_file(file)
                all_chunks.extend(chunks)
            
            if all_chunks:
                engine.build_vector_store(all_chunks)
                st.success(f"Processed {len(uploaded_files)} files into {len(all_chunks)} chunks.")
            else:
                st.warning("No documents to process.")

# Try to load existing vector store
if engine.vector_store is None:
    engine.load_vector_store()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if engine.chain is None:
            st.error("Please process some documents first!")
        else:
            with st.spinner("Thinking..."):
                response = engine.ask(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
