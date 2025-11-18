# AmbedkarGPT-Intern-Task

A simple command-line RAG (Retrieval-Augmented Generation) Q&A system built using:

- LangChain
- ChromaDB (local vector store)
- HuggingFace sentence-transformers (MiniLM)
- Ollama (Mistral 7B local LLM)

This project ingests a text excerpt from Dr. B.R. Ambedkar’s speech and answers questions based ONLY on that content.

---

##  Features

- Loads and splits speech text into chunks  
- Converts chunks into embeddings  
- Stores embeddings in a local Chroma vector database  
- Retrieves relevant context based on user queries  
- Generates answers using Mistral 7B via Ollama  

---

##  Requirements

- Python 3.8+
- Ollama installed locally  
  Download: https://ollama.ai

Then pull the Mistral model:

