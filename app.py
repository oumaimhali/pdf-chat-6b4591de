
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# Configuration
st.title("Chat avec IMEE 2024_0.pdf")

# Initialisation
@st.cache_resource
def init_chatbot():
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.load_local("vectorstore")
    llm = ChatOpenAI(temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return conversation

# Interface de chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

conversation = init_chatbot()

question = st.text_input("Pose ta question sur le PDF :")
if question:
    response = conversation({"question": question, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append({"question": question, "answer": response["answer"]})
    
    for chat in st.session_state.chat_history:
        st.markdown(f"**Q:** {chat['question']}")
        st.markdown(f"**R:** {chat['answer']}")
        st.markdown("---")
