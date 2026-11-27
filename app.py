
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Please configure your .env file."
    )

PDF_URL = (
    "https://media-cdn.cosmofeed.com/chat/"
    "12-Steps-toGen-AI-Expert-2026-14-03-03-24-34.pdf"
)

COLLECTION_NAME = "research_collection"
PERSIST_DIRECTORY = "./chroma_langchain_db"


# ==========================================
# 2. LOAD PDF DOCUMENT
# ==========================================

def load_pdf(file_path):

    print("\nLoading PDF document...")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(f"Successfully loaded {len(documents)} pages.")

    return documents


# ==========================================
# 3. SPLIT DOCUMENT INTO CHUNKS
# ==========================================

def split_documents(documents):

    print("\nSplitting document into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    all_splits = text_splitter.split_documents(documents)

    print(f"Paper split into {len(all_splits)} sub-documents.")

    return all_splits


# ==========================================
# 4. CREATE EMBEDDING MODEL
# ==========================================

def create_embedding_model():

    print("\nLoading embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    return embedding_model


# ==========================================
# 5. CREATE VECTOR DATABASE
# ==========================================

def create_vector_store(all_splits, embedding_model):

    print("\nInitializing Chroma vector database...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )

    if vector_store._collection.count() == 0:

        print("Adding documents to vector database...")

        document_ids = vector_store.add_documents(
            documents=all_splits
        )

        print(f"Stored {len(document_ids)} document chunks.")

    else:
        print("Existing vector database loaded.")

    return vector_store


# ==========================================
# 6. RETRIEVE RELEVANT CONTEXT
# ==========================================

def retrieve_context(query, vector_store, k=6):

    retrieved_docs = vector_store.similarity_search(
        query,
        k=k
    )

    docs_content = ""

    for doc in retrieved_docs:

        docs_content += f"Source: {doc.metadata}\n"
        docs_content += f"Content: {doc.page_content}\n\n"

    return docs_content, retrieved_docs


# ==========================================
# 7. INITIALIZE GEMINI LLM
# ==========================================

def initialize_llm():

    print("\nInitializing Gemini model...")

    model = init_chat_model(
        "google_genai:gemini-3.6-flash",
        api_key=GOOGLE_API_KEY
    )

    return model


# ==========================================
# 8. DOCUCHAT QUESTION ANSWERING
# ==========================================

def docu_chat(user_query, vector_store, model):

    context, source_docs = retrieve_context(
        user_query,
        vector_store,
        k=6
    )

    system_message = f"""
    You are DocuChat, a helpful research paper assistant.

    Answer the user's question using ONLY the provided context.

    Provide clear, structured and easy-to-understand answers.

    Do not make up information.

    If the answer is not available in the context,
    clearly mention that the retrieved context does not
    provide the answer.

    Context:
    {context}
    """

    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    response = model.invoke(messages)

    answer = response.content

    if isinstance(answer, list):

        answer = "\n".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    return {
        "answer": answer,
        "source_document": source_docs,
        "context_used": context
    }


# ==========================================
# 9. MAIN APPLICATION
# ==========================================

def main():

    print("=" * 50)
    print("       DOCUCHAT - RAG ASSISTANT")
    print("=" * 50)

    documents = load_pdf(PDF_URL)

    all_splits = split_documents(documents)

    embedding_model = create_embedding_model()

    vector_store = create_vector_store(
        all_splits,
        embedding_model
    )

    model = initialize_llm()

    print("\nDocuChat is ready!")
    print("Ask questions about your research paper.")
    print("Type 'exit' to stop.\n")

    while True:

        user_query = input("You: ").strip()

        if user_query.lower() in ["exit", "quit"]:
            print("Thank you for using DocuChat!")
            break

        if not user_query:
            continue

        try:

            result = docu_chat(
                user_query,
                vector_store,
                model
            )

            print("\nDocuChat Answer:")
            print(result["answer"])

            print("\nSource Documents:")

            displayed_sources = set()

            for doc in result["source_document"]:

                title = doc.metadata.get("title", "Unknown")
                page = doc.metadata.get("page_label", "Unknown")
                source = doc.metadata.get("source", "")

                if (source, page) not in displayed_sources:

                    print("Title:", title)
                    print("Page:", page)
                    print()

                    displayed_sources.add((source, page))

            print("-" * 50)

        except Exception as error:
            print(f"Error: {error}")


# ==========================================
# 10. EXECUTION
# ==========================================

if __name__ == "__main__":
    main()
