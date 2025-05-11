import os
import argparse
import time
import chromadb
from openai import OpenAI # Updated import
from dotenv import load_dotenv
from pypdf import PdfReader
import re

# --- Configuration ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_SERVER_HOST = os.getenv("CHROMA_SERVER_HOST")
CHROMA_SERVER_HTTP_PORT = os.getenv("CHROMA_SERVER_HTTP_PORT")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "board_game_manuals")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
# Initialize OpenAI client (v1.x.x)
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except TypeError as e: # Handles older versions if someone runs with openai < 1.0 by mistake
    raise ImportError(f"OpenAI library version issue or API key not picked up: {e}. Please ensure openai>=1.0.0 is installed.")


if not CHROMA_SERVER_HOST or not CHROMA_SERVER_HTTP_PORT:
    raise ValueError("Chroma Cloud connection details (HOST, PORT) not set.")

EMBEDDING_MODEL = "text-embedding-ada-002"
TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVERLAP = 200
SUPPORTED_GAMES_FILE = os.path.join("data", "supported_games.txt")

# --- Helper Functions ---
def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    print(f"  Finished extracting text. Total characters: {len(text)}")
    return text

def chunk_text(text, chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=TEXT_CHUNK_OVERLAP):
    print(f"Chunking text (size: {chunk_size}, overlap: {chunk_overlap})...")
    if not text: return []
    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + chunk_size
        chunks.append(text[start_index:end_index])
        start_index += chunk_size - chunk_overlap
    print(f"  Finished chunking. Number of chunks: {len(chunks)}")
    return [chunk for chunk in chunks if chunk.strip()]

def get_embeddings_openai(texts, model=EMBEDDING_MODEL, batch_size=20): # OpenAI API has batch limits for embeddings (e.g. 2048 inputs)
    print(f"Getting embeddings for {len(texts)} text chunks (batch size: {batch_size})...")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        # Ensure batch_texts is not empty
        if not batch_texts:
            continue
        try:
            print(f"  Processing batch {i//batch_size + 1}...")
            response = openai_client.embeddings.create(input=batch_texts, model=model)
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
            print(f"    Got {len(embeddings)} embeddings for this batch.")
            time.sleep(0.5) # Be respectful of rate limits
        except openai_client.RateLimitError: # Updated error type
            print("    Rate limit hit, sleeping for 20 seconds...")
            time.sleep(20)
            # Retry current batch
            response = openai_client.embeddings.create(input=batch_texts, model=model)
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
            print(f"    Got {len(embeddings)} embeddings for this batch (after retry).")
        except Exception as e:
            print(f"    An error occurred during embedding batch {i//batch_size + 1}: {e}")
            # Optionally, decide to skip this batch or stop
            continue 
    print(f"  Finished getting embeddings. Total embeddings: {len(all_embeddings)}")
    return all_embeddings

def clean_game_name(filename):
    """Remove _Manual, _Rule, _manual (case-insensitive) from the filename before formatting."""
    name = os.path.splitext(filename)[0]
    # Remove _Manual, _Rule, _manual (case-insensitive) at the end or before extension
    name = re.sub(r'(_manual|_rule)$', '', name, flags=re.IGNORECASE)
    name = name.replace('_', ' ').replace('-', ' ').title()
    return name

def update_supported_games_list(processed_pdf_names):
    os.makedirs(os.path.dirname(SUPPORTED_GAMES_FILE), exist_ok=True)
    game_names = [clean_game_name(name) for name in processed_pdf_names]
    try:
        with open(SUPPORTED_GAMES_FILE, "w") as f:
            for name in sorted(list(set(game_names))): # Sort and ensure unique
                f.write(name + "\n")
        print(f"Updated supported games list at: {SUPPORTED_GAMES_FILE}")
    except IOError as e:
        print(f"Error writing supported games list: {e}")

# --- Main Ingestion Logic ---
def main(pdf_directory, collection_name_arg, clear_collection=False):
    print("Starting Board Game Manual ingestion process...")
    
    collection_name_to_use = collection_name_arg or DEFAULT_COLLECTION_NAME
    print(f"Target Chroma Collection: {collection_name_to_use}")


    headers = {}
    if CHROMA_API_KEY:
        headers["Authorization"] = f"Bearer {CHROMA_API_KEY}"

    try:
        chroma_db_client = chromadb.HttpClient(
            host=CHROMA_SERVER_HOST, port=CHROMA_SERVER_HTTP_PORT, ssl=True, headers=headers
        )
        chroma_db_client.heartbeat()
        print(f"Successfully connected to Chroma Cloud at {CHROMA_SERVER_HOST}:{CHROMA_SERVER_HTTP_PORT}")
    except Exception as e:
        print(f"Failed to connect to Chroma Cloud: {e}")
        return

    try:
        if clear_collection:
            print(f"Attempting to delete existing collection '{collection_name_to_use}' (if it exists)...")
            try:
                chroma_db_client.delete_collection(name=collection_name_to_use)
                print(f"Collection '{collection_name_to_use}' deleted successfully.")
            except Exception as e:
                print(f"Could not delete collection '{collection_name_to_use}' (may not exist or other error): {e}")
        
        print(f"Getting or creating collection: '{collection_name_to_use}'")
        collection = chroma_db_client.get_or_create_collection(
            name=collection_name_to_use, metadata={"hnsw:space": "cosine"}
        )
        print(f"Collection '{collection_name_to_use}' ready. Current count: {collection.count()}")
    except Exception as e:
        print(f"Error getting or creating collection: {e}")
        return

    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in directory: {pdf_directory}")
        if clear_collection or not os.path.exists(SUPPORTED_GAMES_FILE): # Ensure file is created/cleared if no PDFs
            update_supported_games_list([])
        return
    print(f"Found {len(pdf_files)} PDF files to process: {pdf_files}")

    all_docs, all_embeddings_list, all_metadatas, all_ids = [], [], [], []
    processed_pdf_names = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        doc_text = extract_text_from_pdf(pdf_path)
        if not doc_text.strip():
            print(f"No text extracted from {pdf_file}, skipping.")
            continue
        
        text_chunks = chunk_text(doc_text)
        if not text_chunks:
            print(f"No text chunks generated for {pdf_file}, skipping.")
            continue

        print(f"Generating embeddings for {len(text_chunks)} chunks from {pdf_file}...")
        chunk_embeddings = get_embeddings_openai(text_chunks)

        if len(text_chunks) != len(chunk_embeddings):
            print(f"Mismatch in chunk ({len(text_chunks)}) and embedding ({len(chunk_embeddings)}) count for {pdf_file}. Skipping.")
            continue
        
        game_name_from_file = clean_game_name(pdf_file)
        for i, chunk in enumerate(text_chunks):
            all_docs.append(chunk)
            all_metadatas.append({"source_file": pdf_file, "game_name": game_name_from_file, "chunk_num": i + 1})
            all_ids.append(f"{os.path.splitext(pdf_file)[0]}_chunk_{i+1}")
        
        all_embeddings_list.extend(chunk_embeddings)
        processed_pdf_names.append(pdf_file)
        print(f"Processed {pdf_file}. Added {len(text_chunks)} chunks.")

    if all_docs:
        if not (len(all_docs) == len(all_embeddings_list) == len(all_metadatas) == len(all_ids)):
            print("Error: Mismatch in lengths of documents, embeddings, metadatas, or IDs. Aborting add.")
            return

        print(f"Adding {len(all_docs)} documents to collection '{collection_name_to_use}'...")
        try:
            # ChromaDB client handles batching internally for `add` to some extent,
            # but for very large datasets, manual batching might still be wise.
            # Max items in a single add call is ~41666.
            batch_size_chroma_add = 40000 
            for i in range(0, len(all_docs), batch_size_chroma_add):
                collection.add(
                    documents=all_docs[i:i+batch_size_chroma_add],
                    embeddings=all_embeddings_list[i:i+batch_size_chroma_add],
                    metadatas=all_metadatas[i:i+batch_size_chroma_add],
                    ids=all_ids[i:i+batch_size_chroma_add]
                )
                print(f"  Added batch {i//batch_size_chroma_add + 1} to Chroma.")
            print(f"Successfully added {len(all_docs)} documents to Chroma.")
            print(f"Collection '{collection_name_to_use}' new count: {collection.count()}")
            update_supported_games_list(processed_pdf_names)
        except Exception as e:
            print(f"Error adding documents to Chroma: {e}")
    else:
        print("No documents to add to the collection.")
        if not processed_pdf_names and pdf_files:
             print("No PDFs were successfully processed to update the supported games list.")
        elif not pdf_files : # No PDFs in the directory to begin with
            update_supported_games_list([])

    print("Ingestion process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF board game manuals into Chroma Cloud.")
    parser.add_argument("--pdf_dir", type=str, default="data", help="Directory containing PDF files.")
    parser.add_argument("--collection_name", type=str, default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    parser.add_argument("--clear", action="store_true", help="Clear collection before ingesting.")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_dir):
        os.makedirs(args.pdf_dir)
        print(f"Created directory '{args.pdf_dir}'. Please place PDF manuals here and re-run.")
        exit()
    
    main(args.pdf_dir, args.collection_name, args.clear)