import os
import argparse
import time
import re # For regex preprocessing and game name cleaning
import uuid # For Qdrant point IDs
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
from qdrant_client import QdrantClient, models # Qdrant imports

# --- Configuration ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # May be None if auth not enabled
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "board_game_manuals")

if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY not set.")
if not QDRANT_URL: raise ValueError("QDRANT_URL not set.")

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except TypeError:
    raise ImportError("OpenAI library version issue. Ensure openai>=1.0.0.")

# Qdrant Client Initialization
try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60) # Increased timeout for potentially long operations
    qdrant_client.get_collections() # Test connection
    print(f"Successfully connected to Qdrant at {QDRANT_URL}")
except Exception as e:
    raise ConnectionError(f"Failed to connect to Qdrant: {e}")


EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536 # Dimension for text-embedding-ada-002
TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVERLAP = 200
SUPPORTED_GAMES_FILE = os.path.join("data", "supported_games.txt")
UPSERT_BATCH_SIZE = 100 # Batch size for upserting points to Qdrant

# --- Helper Functions ---

def extract_text_from_pdf(pdf_path):
    # (Same as before)
    print(f"Extracting text from: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n" # Add newline between pages for context
        except Exception as e:
            print(f"  Warning: Could not extract text from page {page_num + 1} of {pdf_path}. Error: {e}")
            continue # Skip problematic pages
    print(f"  Finished extracting text. Total characters: {len(text)}")
    return text

def preprocess_text(text: str) -> str:
    """Cleans raw text extracted from PDF."""
    print("Preprocessing text...")
    # 1. Replace multiple spaces/tabs with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    # 2. Replace multiple newlines with a single newline (optional, might preserve paragraph structure better)
    # text = re.sub(r'\n+', '\n', text).strip()
    # 3. Remove specific known garbage characters or patterns (example)
    # text = text.replace('©', '')
    # 4. Handle hyphenated words at line breaks (optional, complex)
    # text = re.sub(r'(\w)-\n(\w)', r'\1\2', text) # Simplistic approach
    
    # Add more rules as needed based on observed PDF artifacts
    
    # Example: Remove potential headers/footers if they follow a simple pattern
    # like "Page X of Y" - Be careful not to remove actual content!
    # text = re.sub(r'\nPage \d+ of \d+\n', '\n', text, flags=re.IGNORECASE)
    
    print(f"  Preprocessing finished. Length after cleaning: {len(text)}")
    return text

def chunk_text(text, chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=TEXT_CHUNK_OVERLAP):
    # (Same as before)
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

def get_embeddings_openai(texts, model=EMBEDDING_MODEL, batch_size=50): # Adjusted batch size
    # (Same logic as before, using openai_client v1.x)
    print(f"Getting embeddings for {len(texts)} text chunks (batch size: {batch_size})...")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        if not batch_texts: continue
        try:
            print(f"  Processing batch {i//batch_size + 1}...")
            response = openai_client.embeddings.create(input=batch_texts, model=model)
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
            print(f"    Got {len(embeddings)} embeddings for this batch.")
            time.sleep(0.5) 
        except openai_client.RateLimitError:
            print("    Rate limit hit, sleeping for 20 seconds...")
            time.sleep(20); response = openai_client.embeddings.create(input=batch_texts, model=model); embeddings = [item.embedding for item in response.data]; all_embeddings.extend(embeddings)
            print(f"    Got {len(embeddings)} embeddings for batch (after retry).")
        except Exception as e:
            print(f"    An error occurred during embedding batch {i//batch_size + 1}: {e}")
            continue
    print(f"  Finished getting embeddings. Total embeddings: {len(all_embeddings)}")
    return all_embeddings

def clean_game_name(filename):
    """Remove _Manual, _Rule, _Rules (case-insensitive) etc. from the filename before formatting."""
    name = os.path.splitext(filename)[0]
    # Remove common suffixes like _Manual, _Rule(s), variations with/without underscore/dash
    name = re.sub(r'[_ -]?(manual|rulebook|rules|rule)$', '', name, flags=re.IGNORECASE).strip()
    # General replacements and Title Casing
    name = name.replace('_', ' ').replace('-', ' ').strip().title()
    return name

def update_supported_games_list(processed_pdf_names):
    # (Using new clean_game_name function)
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
    print("Starting Board Game Manual ingestion process (using Qdrant)...")
    
    collection_name_to_use = collection_name_arg or QDRANT_COLLECTION_NAME
    print(f"Target Qdrant Collection: {collection_name_to_use}")

    # 1. Ensure Qdrant Collection Exists
    try:
        collection_info = qdrant_client.get_collection(collection_name=collection_name_to_use)
        print(f"Collection '{collection_name_to_use}' already exists.")
        if clear_collection:
            print(f"Recreating collection '{collection_name_to_use}'...")
            qdrant_client.recreate_collection(
                collection_name=collection_name_to_use,
                vectors_config=models.VectorParams(size=EMBEDDING_DIMENSION, distance=models.Distance.COSINE)
            )
            print("Collection recreated.")
    except Exception as e: # Catch specific exception for collection not found if possible, else generic
         if "not found" in str(e).lower() or "status_code=404" in str(e): # Basic check
             print(f"Collection '{collection_name_to_use}' not found. Creating...")
             qdrant_client.create_collection(
                 collection_name=collection_name_to_use,
                 vectors_config=models.VectorParams(size=EMBEDDING_DIMENSION, distance=models.Distance.COSINE)
             )
             print("Collection created.")
         else:
             print(f"Error checking/creating collection: {e}")
             return # Stop if we can't ensure collection exists


    # 2. Process PDFs
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in directory: {pdf_directory}")
        if clear_collection or not os.path.exists(SUPPORTED_GAMES_FILE):
            update_supported_games_list([])
        return
    print(f"Found {len(pdf_files)} PDF files to process: {pdf_files}")

    all_points_to_upsert = []
    processed_pdf_names = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print("-" * 50)
        print(f"Processing file: {pdf_file}")

        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip(): print(f"No text extracted from {pdf_file}, skipping."); continue

        processed_text = preprocess_text(raw_text)
        if not processed_text.strip(): print(f"Text empty after preprocessing {pdf_file}, skipping."); continue

        text_chunks = chunk_text(processed_text)
        if not text_chunks: print(f"No text chunks generated for {pdf_file}, skipping."); continue

        print(f"Generating embeddings for {len(text_chunks)} chunks from {pdf_file}...")
        chunk_embeddings = get_embeddings_openai(text_chunks)

        if len(text_chunks) != len(chunk_embeddings):
            print(f"Mismatch in chunk ({len(text_chunks)}) and embedding ({len(chunk_embeddings)}) count for {pdf_file}. Skipping.")
            continue
        
        cleaned_game_name = clean_game_name(pdf_file)
        print(f"  Cleaned Game Name: {cleaned_game_name}")

        # Prepare Qdrant points for this file
        file_points = []
        for i, chunk in enumerate(text_chunks):
            point_id = str(uuid.uuid4()) # Generate unique ID for each chunk point
            payload = {
                "text": chunk,
                "metadata": { # Nest metadata for clarity
                    "source_file": pdf_file,
                    "game_name": cleaned_game_name,
                    "chunk_num": i + 1,
                    "original_text_length": len(raw_text), # Example of additional metadata
                    "chunk_length": len(chunk)
                }
            }
            
            file_points.append(models.PointStruct(
                id=point_id,
                vector=chunk_embeddings[i],
                payload=payload
            ))
        
        all_points_to_upsert.extend(file_points)
        processed_pdf_names.append(pdf_file)
        print(f"  Prepared {len(file_points)} points for {pdf_file}.")

    # 3. Upsert to Qdrant Collection (in batches)
    if all_points_to_upsert:
        print("-" * 50)
        print(f"Upserting {len(all_points_to_upsert)} total points to collection '{collection_name_to_use}' in batches of {UPSERT_BATCH_SIZE}...")
        try:
            for i in range(0, len(all_points_to_upsert), UPSERT_BATCH_SIZE):
                batch = all_points_to_upsert[i : i + UPSERT_BATCH_SIZE]
                qdrant_client.upsert(
                    collection_name=collection_name_to_use,
                    points=batch,
                    wait=True # Wait for operation to complete
                )
                print(f"  Upserted batch {i//UPSERT_BATCH_SIZE + 1}/{ (len(all_points_to_upsert) + UPSERT_BATCH_SIZE - 1)//UPSERT_BATCH_SIZE }")
            
            print(f"Successfully upserted {len(all_points_to_upsert)} points to Qdrant.")
            
            # Verify count (optional, can take a moment for index to update)
            time.sleep(2) # Give Qdrant a moment to index
            try:
                count_info = qdrant_client.count(collection_name=collection_name_to_use, exact=True)
                print(f"Collection '{collection_name_to_use}' new count: {count_info.count}")
            except Exception as count_e:
                print(f"Could not get exact count after upsert: {count_e}")

            update_supported_games_list(processed_pdf_names)
        except Exception as e:
            print(f"Error upserting documents to Qdrant: {e}")
            # Consider adding more detailed error logging here
    else:
        print("-" * 50)
        print("No documents to add to the collection.")
        if not processed_pdf_names and pdf_files:
             print("No PDFs were successfully processed.")
        elif not pdf_files: # No PDFs in dir
            if clear_collection or not os.path.exists(SUPPORTED_GAMES_FILE):
                update_supported_games_list([])

    print("Ingestion process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF board game manuals into Qdrant Cloud.")
    parser.add_argument("--pdf_dir", type=str, default="data", help="Directory containing PDF files.")
    parser.add_argument("--collection_name", type=str, default=None, help=f"Qdrant collection name (defaults to env var QDRANT_COLLECTION_NAME or '{QDRANT_COLLECTION_NAME}').")
    parser.add_argument("--clear", action="store_true", help="Recreate the Qdrant collection before ingesting. Use with caution!")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_dir):
        os.makedirs(args.pdf_dir)
        print(f"Created directory '{args.pdf_dir}'. Place PDF manuals here and re-run.")
        exit()
    
    main(args.pdf_dir, args.collection_name, args.clear)