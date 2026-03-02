from dotenv import load_dotenv
load_dotenv()

import shutil
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

loader = PyPDFLoader("C:\\git\\docQA\\awspdf.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"{len(chunks)} chunks")

embeddings = OpenAIEmbeddings()

db_path = "./chroma_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_path)
print("Chunks added to vectorstore")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context:\n\n"
    "{context}\n\n"
    "Question: {question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("What are the different services that are explained in this document?")
print(answer)




