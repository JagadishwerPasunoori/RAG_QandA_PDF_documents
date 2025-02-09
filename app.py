import tempfile
import os
import sys
from packaging import version

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import tempfile
import os

# Initialize session state for vector store
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

st.title("📄 RAG Q&A Application with PDF Documents")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("🔑 Enter OpenAI API Key", type="password")
    uploaded_files = st.file_uploader("📤 Upload PDFs", type="pdf", accept_multiple_files=True)
    process_btn = st.button("🔨 Process Documents")

# Document processing section
if process_btn and uploaded_files and api_key:
    with st.spinner("⏳ Processing documents..."):
        all_chunks = []
        
        for uploaded_file in uploaded_files:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name
            
            # Load and split PDF
            loader = PyPDFLoader(temp_path)
            pages = loader.load_and_split()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
            
            # Cleanup temporary file
            os.unlink(temp_path)
        
        # Create vector store
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        st.session_state.vector_store = FAISS.from_documents(all_chunks, embeddings)
        st.success("✅ Documents processed successfully!")

elif process_btn and not api_key:
    st.error("❌ Please provide an OpenAI API key!")
elif process_btn and not uploaded_files:
    st.error("❌ Please upload PDF documents!")

# Question answering section
query = st.text_input("💬 Ask a question about your documents:")
if query:
    if st.session_state.vector_store and api_key:
        # Initialize QA system
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=api_key, temperature=0),
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever()
        )
        
        # Get and display answer
        answer = qa.run(query)
        st.subheader("📝 Answer:")
        st.write(answer)
    else:
        st.warning("⚠️ Please process documents and ensure API key is entered!")
