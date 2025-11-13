import streamlit as st  # Streamlit for web interface
from PyPDF2 import PdfReader  # For extracting text from PDFs
import asyncio  # For async operations
import os  # OS utilities
import logging  # Logging errors and info
from dotenv import load_dotenv  # Load environment variables from .env
import datetime  # For dynamic greetings

from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split text into chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI  # LLM + embeddings
from langchain_community.vectorstores.faiss import FAISS  # FAISS vector store for semantic search
from langchain.chains import LLMChain  # Chain for Q&A
from langchain.prompts import PromptTemplate  # For structured prompts

# Optional OCR (for scanned PDFs)
from pdf2image import convert_from_bytes
import pytesseract


# Setup logging

logging.basicConfig(level=logging.INFO)  # Log info messages
logger = logging.getLogger(__name__)  # Logger instance


# Handle asyncio event loop for Windows

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())  # Create new event loop if none exists


# Load environment variables

load_dotenv()


# Initialize session state

# Session state keeps data across Streamlit reruns
if "test_mode" not in st.session_state:
    st.session_state["test_mode"] = False
if "documentation" not in st.session_state:
    st.session_state["documentation"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # List of Q&A dictionaries
if "user_question" not in st.session_state:
    st.session_state["user_question"] = ""
if "user_api_key" not in st.session_state:
    st.session_state["user_api_key"] = ""
if "api_valid" not in st.session_state:
    st.session_state["api_valid"] = False


# Streamlit page configuration

st.set_page_config(page_title="PDF Chatbot", page_icon="🤖", layout="centered")
st.title("👾 PDF ChatBot Assistant 📚✨")


# Greeting based on current time

hour = datetime.datetime.now().hour
if hour < 12:
    greeting = "🌅 Good Morning! Ready to explore your PDFs?"
elif 12 <= hour < 18:
    greeting = "☀️ Good Afternoon! Let's dive into your PDF content!"
else:
    greeting = "🌙 Good Evening! Ready to get answers from your PDFs?"

st.markdown(f"### {greeting}")
st.markdown("📂 **Upload a PDF file in the left sidebar** and 💡 **ask questions about its content**.👾")


# Sidebar - API Key input

st.sidebar.header("🔑 API Key Setup")
st.session_state["user_api_key"] = st.sidebar.text_input(
    "Enter your Gemini API Key:",
    value=st.session_state["user_api_key"],
    type="password"  # Hide input
)


# Validate API key

if st.sidebar.button("✅ Validate API Key"):
    if not st.session_state["user_api_key"].strip():
        st.sidebar.error("Please enter an API key.")
    else:
        try:
            # Test the API key by making a simple request
            test_client = ChatGoogleGenerativeAI(
                google_api_key=st.session_state["user_api_key"],
                model="gemini-2.5-flash"
            )
            test_client.invoke("Hello")
            st.sidebar.success("API Key is valid! You can now upload PDFs.")
            st.session_state["api_valid"] = True
        except Exception:
            st.sidebar.error("API key is not valid or fake ❌")
            st.session_state["api_valid"] = False


# Sidebar - PDF Upload

if st.session_state.get("api_valid", False):
    uploaded_file = st.sidebar.file_uploader("📂 Upload your PDF", type=["pdf"])
else:
    uploaded_file = None


# Sidebar - Chat history display

with st.sidebar.expander("📝 Chat History", expanded=False):
    if st.session_state.chat_history:
        # Show most recent chat first
        for i, qa in enumerate(reversed(st.session_state.chat_history), 1):
            st.markdown(f"**Q{i}:** {qa['question']}")
            st.markdown(f"**A{i}:** {qa['answer']}")
            st.markdown("---")
    else:
        st.info("No chat history yet.")

# Button to clear chat history
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared.")


# Initialize LLM and embeddings if API valid

if st.session_state.get("api_valid", False):
    try:
        client = ChatGoogleGenerativeAI(
            google_api_key=st.session_state["user_api_key"],
            model="gemini-2.5-flash"
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=st.session_state["user_api_key"],
            model="gemini-embedding-001"
        )
    except Exception as e:
        st.error(f"Google GenAI init failed: {e}")
        st.stop()
else:
    client = None
    embeddings = None


# Prompt template for structured Q&A

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer step by step in a structured Q&A format:\n"
        "Q1: ...\nA1: ...\n"
        "Q2: ...\nA2: ...\n"
        "If only one answer is needed, still reply as Q1/A1."
    )
)
qa_chain = LLMChain(llm=client, prompt=prompt_template) if client else None


# Function: Process PDF

def process_pdf(file):
    """
    Extract text from PDF. If no text, apply OCR. 
    Split text into chunks and store in FAISS vector store.
    """
    try:
        with st.spinner("Processing PDF..."):
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

            # If no text, try OCR
            if not text.strip():
                st.warning("No extractable text found, trying OCR...")
                images = convert_from_bytes(file.read())
                for img in images:
                    text += pytesseract.image_to_string(img)

            if not text.strip():
                raise ValueError("No text extracted from PDF.")

            # Split text into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_text(text)
            if not texts:
                raise ValueError("Text splitting failed.")

            # Create FAISS vector store for semantic search
            vector_store = FAISS.from_texts(texts, embedding=embeddings)
            vector_store.save_local("faiss_index")
            logger.info(f"Processed PDF with {len(texts)} chunks.")
            return vector_store
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        return None


# Function: Answer Question

def answer_question(vector_store, question):
    """
    Search vector store for relevant chunks and get structured answer from LLM.
    """
    try:
        # Find top 3 similar text chunks
        docs = vector_store.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        # Generate answer using QA chain
        answer = qa_chain.run(context=context, question=question)
        # Save Q&A in session state
        st.session_state.chat_history.append({"question": question, "answer": answer})
        return answer
    except Exception as e:
        logger.error(f"QA error: {e}")
        return None


# Main App Logic

if st.session_state.get("api_valid", False) and uploaded_file:
    if not uploaded_file.name.endswith(".pdf"):
        st.error("Only PDF files are supported.")
    elif uploaded_file.size == 0:
        st.error("Uploaded PDF is empty.")
    elif uploaded_file.size > 10 * 1024 * 1024:
        st.error("File too large (max 10MB).")
    else:
        # Process PDF
        vector_store = process_pdf(uploaded_file)
        if not vector_store:
            st.error("Could not process PDF.")
        else:
            st.success("PDF processed successfully! ✅")

            # Input form for user question
            with st.form("question_form", clear_on_submit=True):
                question = st.text_input("💡 Ask a question about the PDF:", key="user_question")
                submitted = st.form_submit_button("Ask")

            # Get answer if question submitted
            if submitted and question.strip():
                with st.spinner("Finding answer..."):
                    answer = answer_question(vector_store, question)
                if not answer:
                    st.error("Could not answer. Try rephrasing.")

            # Display chat in bubble style
            for qa in reversed(st.session_state.chat_history):
                st.markdown(
                    f"<div style='background-color:#d1e7dd; padding:10px; border-radius:10px; "
                    f"margin:5px; text-align:right; max-width:70%; float:right;'>🧑 {qa['question']}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div style='background-color:#f0f2f6; padding:10px; border-radius:10px; "
                    f"margin:5px; text-align:left; max-width:70%; float:left;'>🤖 {qa['answer']}</div>",
                    unsafe_allow_html=True
                )
                st.markdown("<div style='clear:both;'></div>", unsafe_allow_html=True)

else:
    if not st.session_state.get("api_valid", False):
        st.info("🔒 Please enter and validate your API key in the sidebar to unlock PDF upload.")
    elif not uploaded_file:
        st.info("📂 Please upload a PDF file.")


# Optional Documentation

if st.session_state.get("documentation", False):
    with st.expander("📘 Documentation"):
        st.markdown("""
        ### PDF Chatbot - How to Use
        1. Enter a valid **Gemini API key** in the sidebar and validate it.
        2. Upload a **PDF file** using the sidebar.
        3. Ask questions in the input box.
        4. Answers will always be shown in **Q&A format**.
        5. Input clears automatically after pressing Enter.

        **Limitations:**
        - Image-based PDFs require OCR (enabled here).
        - Large PDFs may take longer to process.
        """)


# Test mode banner

if st.session_state.get("test_mode", False):
    st.warning("⚠️ Test mode is enabled.")
