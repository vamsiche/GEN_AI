import streamlit as st
from pypdf import PdfReader
import asyncio
import os
import logging
from dotenv import load_dotenv
import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pdf2image import convert_from_bytes
import pytesseract


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Asyncio fix (Windows / Streamlit)
# --------------------------------------------------

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# --------------------------------------------------
# Environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Streamlit session state
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

if "api_valid" not in st.session_state:
    st.session_state.api_valid = False


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("PDF ChatBot Assistant")


# --------------------------------------------------
# Greeting
# --------------------------------------------------

hour = datetime.datetime.now().hour
if hour < 12:
    greeting = "Good Morning! Ready to explore your PDFs?"
elif hour < 18:
    greeting = "Good Afternoon! Let's dive into your PDFs."
else:
    greeting = "Good Evening! Ask questions from your PDFs."

st.markdown(f"### {greeting}")
st.markdown("Upload a PDF and ask questions about its content.")


# --------------------------------------------------
# Sidebar — API Key
# --------------------------------------------------

st.sidebar.header("API Key Setup")

st.session_state.user_api_key = st.sidebar.text_input(
    "Enter your Gemini API Key",
    type="password",
    value=st.session_state.user_api_key
)

if st.sidebar.button("Validate API Key"):
    if not st.session_state.user_api_key.strip():
        st.sidebar.error("API key is required.")
    else:
        try:
            test_client = ChatGoogleGenerativeAI(
                google_api_key=st.session_state.user_api_key,
                model="gemini-2.5-flash"
            )
            test_client.invoke("Hello")
            st.session_state.api_valid = True
            st.sidebar.success("API key is valid.")
        except Exception:
            st.session_state.api_valid = False
            st.sidebar.error("Invalid API key.")


# --------------------------------------------------
# Sidebar — PDF upload
# --------------------------------------------------

uploaded_file = None
if st.session_state.api_valid:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )


# --------------------------------------------------
# Initialize Gemini + embeddings
# --------------------------------------------------

if st.session_state.api_valid:
    try:
        llm = ChatGoogleGenerativeAI(
            google_api_key=st.session_state.user_api_key,
            model="gemini-2.5-flash"
        )

        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=st.session_state.user_api_key,
            model="gemini-embedding-001"
        )
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
        st.stop()
else:
    llm = None
    embeddings = None


# --------------------------------------------------
# Prompt + Runnable chain
# --------------------------------------------------

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""
)

qa_chain = (
    prompt
    | llm
    | StrOutputParser()
) if llm else None


# --------------------------------------------------
# PDF processing
# --------------------------------------------------

def process_pdf(file):
    try:
        with st.spinner("Processing PDF..."):
            reader = PdfReader(file)
            text = ""

            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

            if not text.strip():
                st.warning("No text found, running OCR...")
                images = convert_from_bytes(file.read())
                for img in images:
                    text += pytesseract.image_to_string(img)

            if not text.strip():
                raise ValueError("No text could be extracted.")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_text(text)

            vector_store = FAISS.from_texts(
                chunks,
                embedding=embeddings
            )

            return vector_store

    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        return None


# --------------------------------------------------
# Question answering
# --------------------------------------------------

def answer_question(vector_store, question):
    try:
        docs = vector_store.similarity_search(question, k=3)
        context = "\n".join(doc.page_content for doc in docs)

        answer = qa_chain.invoke({
            "context": context,
            "question": question
        })

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

        return answer

    except Exception as e:
        logger.error(f"QA error: {e}")
        return None


# --------------------------------------------------
# Main logic
# --------------------------------------------------

if st.session_state.api_valid and uploaded_file:

    vector_store = process_pdf(uploaded_file)

    if vector_store:
        st.success("PDF processed successfully.")

        with st.form("question_form", clear_on_submit=True):
            question = st.text_input("Ask a question about the PDF")
            submitted = st.form_submit_button("Ask")

        if submitted and question.strip():
            with st.spinner("Generating answer..."):
                answer_question(vector_store, question)

        for qa in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {qa['question']}")
            st.markdown(f"**Bot:** {qa['answer']}")
            st.markdown("---")

    else:
        st.error("Failed to process PDF.")

else:
    st.info("Enter API key and upload a PDF to begin.")


