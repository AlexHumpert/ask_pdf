import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains import RetrievalQAWithSourcesChain
from pdfminer.high_level import extract_pages
import requests
import os
from langchain.llms import OpenAI

# Create directory if it doesn't exist
UPLOAD_DIRECTORY = "uploaded_pdfs"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

def generate_response(file_path, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        raw_documents = PyPDFLoader(file_path).load()
        # Define text splitter  
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
        # Splitting documents into chunks
        documents = text_splitter.split_documents(raw_documents)
        # Creating vector store
        db = Chroma.from_documents(documents, OpenAIEmbeddings(openai_api_key=openai_api_key))
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key),retriever=retriever)
        return qa.run(query_text)


# File upload
uploaded_file = st.file_uploader('Upload an article')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            file_path = save_uploaded_file(uploaded_file)
            response = generate_response(file_path, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
