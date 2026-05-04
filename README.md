# 🤖 RAG Chatbot using Gemini + Streamlit

A RAG-based chatbot that improves answer accuracy by retrieving context from uploaded documents. Uses FAISS for similarity search and Google Gemini for response generation, with a Streamlit interface for real-time, grounded, and reliable conversational output.

## 👨‍💻 About this project

This is a simple AI-based PDF chatbot built using **Streamlit** and **Google Gemini API**.

You can upload a PDF and ask questions about its content.
The chatbot reads the PDF, processes it, and gives answers based on the document.

---

## 🚀 Features

* 📂 Upload any PDF file
* 💬 Ask questions about the PDF
* 🧠 Uses Gemini LLM for answering
* 🔍 Semantic search using FAISS
* 🧾 Chat history saved in session
* 🔐 API key validation before usage

---

## 🧰 Tech Stack

* **Streamlit** → UI
* **LangChain** → LLM pipeline
* **Google Gemini API** → AI model
* **FAISS** → vector search
* **PyPDF** → PDF reading

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If no requirements file:

```bash
pip install streamlit pypdf langchain langchain-google-genai faiss-cpu python-dotenv
```

---

### 3. Add API Key

Create a `.env` file:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

---

## ▶️ Run the app

```bash
streamlit run app.py
```

---

## 🧪 How it works

1. Upload a PDF
2. Text is extracted from PDF
3. Text is split into chunks
4. Embeddings are created using Gemini
5. Stored in FAISS vector database
6. User question → similarity search
7. Relevant context → sent to LLM
8. LLM generates answer

---

## 📌 Example Flow

* Upload: `machine_learning.pdf`
* Ask: *"What is supervised learning?"*
* Output: Structured Q&A response

---

## ⚠️ Limitations

* ❌ Scanned PDFs (images) will not work (no OCR)
* ❌ Needs internet connection
* ❌ API key required
* ❌ Large PDFs may be slow

---

## 🧠 What I learned

* Working with LLMs using LangChain
* Vector databases (FAISS)
* Streamlit UI development
* Handling user sessions
* Prompt engineering basics

---

## 🚧 Future Improvements

* Add OCR support for scanned PDFs
* Add file caching (avoid reprocessing)
* Improve UI/UX
* Deploy on cloud (Streamlit Cloud / Render)
* Add multi-PDF support

---

## 📎 File Reference

Main application code is available here:


---

## ⚠️ Note

This project is for learning purposes.
It may not give perfect answers for all PDFs.

---

## ⭐ If you like this

Feel free to fork and improve 🙂
