import streamlit as st
from data_fetch_retrieve_invoke import  answer_question

st.title("Tech Ups Ai")
st.markdown(
    """
    **Welcome to Tech Ups AI**
    An application that allows you to ask questions about recent tech news.
    The app uses a vector store to retrieve relevant articles and answer your questions.
    """
)
st.subheader("Ask a question about any recent tech news:")
question = st.text_input("Enter your question here:")
if question:
    st.write(f"You asked: {question}")
    answer = answer_question(question)
    st.write(f"{answer}")
