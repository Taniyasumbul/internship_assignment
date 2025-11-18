from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama


def build_rag_pipeline():
    # Load text file
    loader = TextLoader("speech.txt")
    documents = loader.load()

    # Split into smaller chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Chroma database
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="db")
    vectordb.persist()

    # Create retriever
    retriever = vectordb.as_retriever()

    # Load Ollama LLM (Mistral)
    llm = Ollama(model="mistral")

    return retriever, llm


if __name__ == "__main__":
    print("\n🧠 AmbedkarGPT Ready!")

    retriever, llm = build_rag_pipeline()

    while True:
        query = input("\nAsk a question (or 'exit'): ")

        if query.lower() == "exit":
            print("👋 Goodbye!")
            break

        # ✅ FIXED: Use retriever.invoke() instead of get_relevant_documents()
        results = retriever.invoke(query)

        # Combine retrieved context
        context = "\n\n".join([r.page_content for r in results])

        # Build prompt for LLM
        prompt = f"""
You are an assistant answering based ONLY on the following context.

Context:
{context}

Question: {query}

Answer:
"""

        # Query Ollama model
        answer = llm.invoke(prompt)
        print("\nAnswer:", answer)
