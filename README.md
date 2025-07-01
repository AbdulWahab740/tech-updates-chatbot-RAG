# 📰 Tech Ups Ai — Your Tech News RAG Chatbot

**(LIVE LINK 🔗 :- https://tech-ups-ai.streamlit.app/)**
***(Note: I am using a free API key for this which may got expired I am sharing some working pics)***

Tech Ups Ai is an intelligent chatbot that answers questions based on the latest tech news using Retrieval-Augmented Generation (RAG). It fetches news from top tech RSS feeds, stores them in a FAISS vector store, and responds using LLMs (Groq + Llama models).

# Features:

🔍 Retrieves and chunks tech news articles with metadata (title, date, URL).

🤖 Uses all-MiniLM-L6-v2 for embedding news content.

📦 Stores chunks in a local FAISS vector store.

🧠 Answers queries using Groq-hosted LLMs.

🖥️ Easy Streamlit interface to interact with the chatbot.

## ⚙️ Setup Instructions

**1. 🧠 Clone the Repository**
`
git clone https://github.com/your-username/newsgpt.git
`
`
cd newsgpt
`
**2. 🐍 Create Environment & Install Requirements**
`python -m venv venv`
`source venv/bin/activate   # or venv\Scripts\activate on Windows`
`pip install -r requirements.txt`

**3. 🔐 Set Up Your .env File**

Create a .env file and paste your Groq API key:

`GROQ_API_KEY=your_groq_key_here`

**💬 Running the Chatbot**

`streamlit run app.py`

Then open: **http://localhost:8501**

**🔍 Example Sources Used**


*TechCrunch*
*The Verge*
*Ars Technica*

🧠 **Models Used**

*all-MiniLM-L6-v2 for embeddings (via HuggingFace)*
*deepseek-r1-distill-llama-70b via Groq LLMs (changeable)*

🛡️ **Disclaimer**
This is an educational, demo-level chatbot. Always verify information before taking action based on it.
