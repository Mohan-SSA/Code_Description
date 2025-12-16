# 💡 Code Description: AI Codebase Explainer & QA
Code Description is a powerful Streamlit application that uses advanced Retrieval-Augmented Generation (RAG) technology, powered by Groq and ChromaDB, to help developers instantly understand and interact with any unfamiliar codebase.

Upload your project, and RepoGuru will generate a high-level overview and provide an interactive chatbot grounded strictly in your source code.

## ✨ Features
- Zero-Setup Code Analysis: Instantly analyze projects without complex local configuration.
- Flexible Ingestion: Upload your code via three methods:
- Multiple File Upload: Select and upload individual files from your project.
- GitHub URL: Clone and analyze public repositories.
- ZIP Upload: Upload a compressed project archive.
- AI-Powered Project Overview: Automatically generates a comprehensive summary of the codebase, identifying key components, files, and their interactions.
- Interactive Code Q&A: Ask specific questions about functions, logic, or dependencies, with answers sourced directly from the indexed code for reliable grounding.
- High-Speed Response: Leverages the Groq API (llama-3.1-8b-instant) for near-instantaneous analysis and chat performance.

⚙️ Technology Stack
Frontend: Streamlit
RAG Engine: ChromaDB (Vector Store) + Groq (LLM)
Code Chunking: langchain-text-splitters

🚀 Setup and Run
1. Prerequisites
You must have Python installed and a Groq API key.
Python: Python 3.9+
Groq API Key: Obtain an API key from Groq and set it up as an environment variable.

2. Installation
Install all the necessary Python packages.
```
# Assuming you have cloned your repository
pip install streamlit chromadb python-dotenv groq
# Also required for code splitting/embedding functions:
pip install langchain-text-splitters chromadb-tool-sentence-transformers
```
3. Configure Environment
Create a file named .env in the same directory as app.py and add your Groq API key:
```
# .env file
GROQ_API_KEY="your_groq_api_key_here"
```
4. Run the Application
Start the Streamlit application from your terminal:
```
streamlit run app.py
```

👨‍💻 Usage
- Select Source Type in the left sidebar: Choose File upload, GitHub repo URL, or ZIP upload.
- Provide Input: Input your files, URL, or ZIP archive.
- Click "Analyse": The app will index the codebase, and the Project Overview will populate with an AI-generated summary.
- Chat: In the "Chat with the Codebase" section, ask any question about the project's structure or logic. The assistant will answer by referencing the code logic and citing the relevant file
