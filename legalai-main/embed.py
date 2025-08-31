import utils as Utils
import os as OS
from tqdm import tqdm
import requests
import fitz  # PyMuPDF
from chromadb.utils import embedding_functions
import chromadb
from dotenv import load_dotenv

load_dotenv()

def pdf_to_text(url):
    try:
        response = requests.get(url)
        pdf_data = response.content
        document = fitz.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def split_text_into_sections(text, min_chars_per_section):
    paragraphs = text.split('\n')
    sections = []
    current_section = ""
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_length + paragraph_length + 2 <= min_chars_per_section:  # +2 for the double newline
            current_section += paragraph + '\n\n'
            current_length += paragraph_length + 2  # +2 for the double newline
        else:
            if current_section:
                sections.append(current_section.strip())
            current_section = paragraph + '\n\n'
            current_length = paragraph_length + 2  # +2 for the double newline

    if current_section:  # Add the last section
        sections.append(current_section.strip())

    return sections

def embed_text_in_chromadb(
    text,
    document_name,
    document_description,
    persist_directory=Utils.DB_FOLDER,
):
    """
    Embed text using Google Gemini (Generative AI) embeddings and store in ChromaDB.

    Requirements:
      - Environment variable GOOGLE_API_KEY must be set.
      - Package google-generativeai must be installed (pulled in by Chroma's embedding function if needed).
    """
    google_api_key = OS.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please export your Google Generative AI API key."
        )

    # Google Gemini embeddings via Chroma's embedding function
    # Model name aligns with Google Generative AI embeddings
    gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=google_api_key,
        model_name="models/text-embedding-004",
    )

    # Split into manageable chunks
    documents = split_text_into_sections(text, 1000)

    # Metadata for the documents
    metadata = {
        "name": document_name,
        "description": document_description,
    }
    metadatas = [metadata] * len(documents)  # Duplicate metadata for each chunk

    # Persist to local ChromaDB
    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = "collection_1"
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=gemini_ef
    )

    # create ids from the current count
    count = collection.count()
    print(f"Collection already contains {count} documents")
    ids = [str(i) for i in range(count, count + len(documents))]

    # load the documents in batches of 100
    for i in tqdm(range(0, len(documents), 100), desc="Adding documents", unit_scale=100):
        collection.add(
            ids=ids[i : i + 100],
            documents=documents[i : i + 100],
            metadatas=metadatas[i : i + 100],  # type: ignore
        )

    new_count = collection.count()
    print(f"Added {new_count - count} documents")
   
if __name__ == "__main__":
    # pdf_path = "TA-9-2024-0138_EN.pdf"
    document_name = "NITI ayoug National Strategy for Artificial Intelligence"
    document_description = "Artificial Intelligence Act"
    text = pdf_to_text(Utils.NITI_AYOG_URL)
    embed_text_in_chromadb(text, document_name, document_description)

