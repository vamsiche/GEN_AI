# RAG Chatbot using Gemini and Streamlit

## About

This project is a PDF-based question answering system built using Streamlit and Google Gemini.

Users can upload a PDF and ask questions about its content. The system processes the document and generates answers using a language model.

---

## Features

* Upload and process PDF files
* Ask questions based on document content
* Uses semantic search for better accuracy
* Maintains session-based chat history
* API key validation before usage

---

## Tech Stack

* Streamlit (Frontend)
* LangChain (LLM pipeline)
* Google Gemini API
* FAISS (Vector database)
* PyPDF (PDF parsing)

---

## Setup

### Install dependencies

```bash
pip install streamlit pypdf langchain langchain-google-genai faiss-cpu python-dotenv
```

---

### Configure environment variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key
```

---

## Run the application

```bash
streamlit run app.py
```

---

## How it works

1. Extracts text from PDF
2. Splits text into chunks
3. Generates embeddings using Gemini
4. Stores embeddings in FAISS
5. Matches user query with relevant content
6. Sends context to LLM for answer generation

---

## Limitations

* Does not support scanned PDFs (no OCR)
* Requires internet connection
* Performance may degrade with large files

---

## Learning Outcomes

* Integration of LLMs with applications
* Vector similarity search
* Streamlit-based UI development
* Prompt engineering basics

---

## Improvements

* Add OCR support
* Cache embeddings to avoid recomputation
* Improve UI/UX
* Support multiple PDFs

---

## Source Code

Main application file:
