import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tempfile

st.title("Document QA")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if uploaded_file:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("file_id") != file_id:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        with st.spinner("Processing file..."):
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            os.unlink(temp_file_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

            if st.session_state.vectorstore is not None:
                st.session_state.vectorstore.delete_collection()
                st.session_state.vectorstore = None

            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            prompt = ChatPromptTemplate.from_template(
                "Answer the question based only on the following context and if you don't know the answer, just give a generic answer:\n\n"
                "{context}\n\n"
                "Question: {question}"
            )
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            st.session_state.rag_chain = rag_chain
            st.session_state.vectorstore = vectorstore
            st.session_state.file_id = file_id
            st.success("File processed successfully!")

    if st.session_state.rag_chain:
        question = st.text_input("Ask a question about the document")
        if question:
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_chain.invoke(question)
            st.write(answer)