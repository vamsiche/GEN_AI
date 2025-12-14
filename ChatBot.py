import streamlit as st
from pypdf import PdfReader
import asyncio
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


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Asyncio fix
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# Load env vars
load_dotenv()


# Session state
if "test_mode" not in st.session_state:
    st.session_state.test_mode = False
if "documentation" not in st.session_state:
    st.session_state.documentation = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_question" not in st.session_state:
    st.session_state.user_question = ""
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""
if "api_valid" not in st.session_state:
    st.session_state.api_valid = False


# Page config
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="🤖",
    layout="centered"
)
st.title("👾 PDF ChatBot Assistant 📚✨")


# Greeting
hour = datetime.datetime.now().hour
if hour < 12:
    greeting = "🌅 Good Morning! Ready to explore your PDFs?"
elif hour < 18:
    greeting = "☀️ Good Afternoon! Let's dive into your PDF content!"
else:
    greeting = "🌙 Good Evening! Ready to get answers from your PDFs?"

st.markdown(f"### {greeting}")
st.markdown("📂 **Upload a PDF file in the left sidebar** and 💡 **ask questions about its content**.👾")


# Sidebar – API key
st.sidebar.header("🔑 API Key Setup")
st.session_state.user_api_key = st.sidebar.text_input(
    "Enter your Gemini API Key:",
    value=st.session_state.user_api_key,
    type="password"
)

if st.sidebar.button("✅ Validate API Key"):
    if not st.session_state.user_api_key.strip():
        st.sidebar.error("Please enter an API key.")
    else:
        try:
            test_client = ChatGoogleGenerativeAI(
                google_api_key=st.session_state.user_api_key,
                model="gemini-2.5-flash"
            )
            test_client.invoke("Hello")
            st.session_state.api_valid = True
            st.sidebar.success("API Key is valid! You can now upload PDFs.")
        except Exception:
            st.session_state.api_valid = False
            st.sidebar.error("API key is not valid ❌")


# Sidebar – PDF upload
uploaded_file = (
    st.sidebar.file_uploader("📂 Upload your PDF", type=["pdf"])
    if st.session_state.api_valid else None
)


# Sidebar – Chat history
with st.sidebar.expander("📝 Chat History", expanded=False):
    if st.session_state.chat_history:
        for i, qa in enumerate(reversed(st.session_state.chat_history), 1):
            st.markdown(f"**Q{i}:** {qa['question']}")
            st.markdown(f"**A{i}:** {qa['answer']}")
            st.markdown("---")
    else:
        st.info("No chat history yet.")

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared.")


# LLM + embeddings
if st.session_state.api_valid:
    llm = ChatGoogleGenerativeAI(
        google_api_key=st.session_state.user_api_key,
        model="gemini-2.5-flash"
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=st.session_state.user_api_key,
        model="gemini-embedding-001"
    )
else:
    llm = None
    embeddings = None


# Prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer step by step in a structured Q&A format:\n"
        "Q1: ...\nA1: ...\n"
        "If only one answer is needed, still reply as Q1/A1."
    )
)

qa_chain = (
    prompt
    | llm
    | StrOutputParser()
) if llm else None


# PDF processing
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
                st.error("❌ This PDF is scanned. OCR is disabled.")
                return None

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = splitter.split_text(text)

            return FAISS.from_texts(texts, embedding=embeddings)

    except Exception as e:
        logger.error(e)
        return None


# Answer question
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
        logger.error(e)
        return None


# Main app
if st.session_state.api_valid and uploaded_file:
    vector_store = process_pdf(uploaded_file)

    if vector_store:
        st.success("PDF processed successfully! ✅")

        with st.form("question_form", clear_on_submit=True):
            question = st.text_input("💡 Ask a question about the PDF:")
            submitted = st.form_submit_button("Ask")

        if submitted and question.strip():
            with st.spinner("Finding answer..."):
                answer_question(vector_store, question)

        for qa in reversed(st.session_state.chat_history):
            st.markdown(
                f"""
                <div style="
                    background-color:#0d6efd;
                    color:#ffffff;
                    padding:12px;
                    border-radius:12px;
                    margin:5px;
                    text-align:right;
                    max-width:70%;
                    float:right;
                ">
                    🧑 {qa['question']}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style="
                    background-color:#1e1e1e;
                    color:#ffffff;
                    padding:12px;
                    border-radius:12px;
                    margin:5px;
                    text-align:left;
                    max-width:70%;
                    float:left;
                ">
                    🤖 {qa['answer']}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div style='clear:both;'></div>", unsafe_allow_html=True)
else:
    st.info("🔒 Please enter and validate your API key in the sidebar to unlock PDF upload.")
