import streamlit as st
from data_fetch_retrieve_invoke import get_tech_news_documents, answer_question


st.title("Tech News Updates GPT")
st.markdown(
    """
    This is a simple NewsGPT application that allows you to ask questions about recent tech news.
    The app uses a vector store to retrieve relevant articles and answer your questions.
    """
)
# Has to send the input to the data_fetch_vectorstore.ipynb file
st.subheader("Ask a question about recent tech news:")
question = st.text_input("Enter your question here:")

if question:
    st.write(f"You asked: {question}")

    answer = answer_question(question)
    st.write(f"Answer: {answer}")