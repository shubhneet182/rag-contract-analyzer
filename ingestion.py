# Import libraries
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time

# Load environment variables from .env file
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

PDF_PATH = "./data/essay_Part4.pdf"
CHROMA_PATH = "chroma_db"

# Load PDF and extract text
def load_pdf(pdf_path):
    print(f"Loading Document: {pdf_path}")
    # PDF Loader
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")
    return docs


# Text Splitter
def split_documents(docs):
    print(f"\nSplitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 150,
        length_function = len,
        separators = ["\n\n","\n","."," ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} pages")
    return chunks

# Embedding Model
def create_embeddings():
    print("Initializing Gemini embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model = "models/gemini-embedding-2"
    )
    print(f"Embedding model ready")
    return embeddings

# Store in Vector DB (ChromaDB)
def store_in_chromadb(chunks, embeddings):
    print(f"\nStoring {len(chunks)} chunks in ChromaDB...")
    batch_size = 20
    total_batches = len(chunks) // batch_size - 1
    print(f"Processing batch 1 of {total_batches}...")
    db = Chroma.from_documents(
        documents = chunks[:batch_size],
        embedding = embeddings,
        persist_directory = CHROMA_PATH
    )
    
    for i in range(batch_size, len(chunks), batch_size):
        batch_num = (i // batch_size) + 1
        print(f"Processing batch {batch_num} of {total_batches}...")
        batch = chunks[i:i + batch_size]
        db.add_documents(batch)
        time.sleep(60)
        
    print(f"Successfully stored in {CHROMA_PATH}")
    return db


if __name__ == "__main__":
    documents = load_pdf(PDF_PATH)
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    db = store_in_chromadb(chunks, embeddings)
    print(f"\nIngestion complete!")
    print(f"\nYour vector database is ready in {CHROMA_PATH}/")
