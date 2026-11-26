# DocuChat – RAG-Powered Research Paper Q&A Assistant

DocuChat is a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about research papers in natural language. It retrieves relevant information from PDF documents using semantic search and generates context-aware answers using Google's Gemini model.

## Features

- **PDF Document Loading:** Loads research papers directly from PDF URLs.
- **Text Chunking:** Splits documents into smaller chunks for efficient retrieval.
- **Semantic Embeddings:** Uses Hugging Face sentence-transformers to generate embeddings.
- **Vector Database:** Stores document embeddings using ChromaDB.
- **Contextual Retrieval:** Retrieves relevant document sections based on user queries.
- **AI-Powered Answers:** Uses Google Gemini to generate answers based on retrieved context.
- **Source Tracking:** Displays source document metadata for retrieved information.
- **Interactive Q&A:** Supports continuous questions through a command-line interface.

## Tech Stack

- Python
- LangChain
- Google Gemini API
- Hugging Face Sentence Transformers
- ChromaDB
- PyPDF

## Project Architecture

1. Load the research paper from a PDF URL.
2. Extract document content using PyPDFLoader.
3. Split the extracted content into chunks.
4. Generate vector embeddings using Hugging Face.
5. Store embeddings in ChromaDB.
6. Retrieve relevant chunks using semantic similarity search.
7. Pass retrieved context and the user query to Gemini.
8. Generate and display a context-aware answer with source metadata.

## Project Structure

```text
DocuChat-RAG-Assistant/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd DocuChat-RAG-Assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.env` file in the project directory and add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_actual_api_key
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 5. Run the Application

```bash
python app.py
```

## Example Usage

```text
==================================================
       DOCUCHAT - RAG ASSISTANT
==================================================

DocuChat is ready!

You: What is Generative AI?

DocuChat Answer:
[Answer generated using the retrieved document context]

Source Documents:
[Retrieved document metadata]

You: exit

Thank you for using DocuChat!
```

## Key Concepts Used

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Semantic Search
- Text Embeddings
- Vector Databases
- Document Retrieval
- Prompt Engineering

## Future Enhancements

- Support uploading multiple PDF documents.
- Develop a Streamlit or React-based user interface.
- Display detailed page-level source citations.
- Add conversation memory.
- Support document summarization.

## Author

**Sai Chandu Dammagalla**

- [GitHub](https://github.com/chandu-4e4)
- [LinkedIn](https://www.linkedin.com/in/saichandudammagalla/)