import streamlit as st
from ChatBot import chatbot_main

st.title("GEN AI Chatbot")
query = st.text_input("Enter your question:")
if st.button("Ask"):
    if query:
        response = chatbot_main(query)
        st.write(response)
    else:
        st.warning("Please enter something.")

