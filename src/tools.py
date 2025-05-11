import os
import json
import chromadb
from openai import OpenAI # Updated import
from dotenv import load_dotenv
from boardgamegeek2 import BGGClient
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

# --- Environment and Clients ---
load_dotenv() # Load .env from root for local execution
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_SERVER_HOST = os.getenv("CHROMA_SERVER_HOST")
CHROMA_SERVER_HTTP_PORT = os.getenv("CHROMA_SERVER_HTTP_PORT")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "board_game_manuals")
EMBEDDING_MODEL = "text-embedding-ada-002"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set.")
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except TypeError:
    raise ImportError("OpenAI library version issue or API key not picked up. Please ensure openai>=1.0.0 is installed.")


_chroma_db_client = None
_bgg_client = None

def get_chroma_client_singleton():
    global _chroma_db_client
    if _chroma_db_client is None:
        if not CHROMA_SERVER_HOST or not CHROMA_SERVER_HTTP_PORT:
            raise ValueError("Chroma connection details missing.")
        headers = {}
        if CHROMA_API_KEY:
            headers["Authorization"] = f"Bearer {CHROMA_API_KEY}"
        _chroma_db_client = chromadb.HttpClient(
            host=CHROMA_SERVER_HOST, port=CHROMA_SERVER_HTTP_PORT, ssl=True, headers=headers
        )
        try:
            _chroma_db_client.heartbeat()
        except Exception as e:
            _chroma_db_client = None
            raise ConnectionError(f"Failed to connect to ChromaDB: {e}")
    return _chroma_db_client

def get_bgg_client_singleton():
    global _bgg_client
    if _bgg_client is None:
        _bgg_client = BGGClient()
    return _bgg_client

# --- Tool Definitions ---

class ChromaSearchInput(BaseModel):
    query: str = Field(description="The question or topic to search for in the board game manuals.")
    game_name: str = Field(None, description="Optional: The specific game name to filter search results for, if known (e.g., 'Catan', 'Ticket to Ride').")
    top_k: int = Field(3, description="Number of relevant chunks to retrieve.")

@tool("search_board_game_manuals", args_schema=ChromaSearchInput)
def search_board_game_manuals(query: str, game_name: str = None, top_k: int = 3) -> str:
    """
    Searches the ingested board game manuals for information relevant to the query.
    Use this to answer specific rule questions or find information within specific game manuals.
    If a specific game is mentioned (e.g., 'Catan', 'Ticket to Ride'), provide the game_name to filter.
    """
    print(f"Tool: search_board_game_manuals called with query: '{query}', game: '{game_name}', top_k: {top_k}")
    try:
        client = get_chroma_client_singleton()
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

        response = openai_client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
        query_embedding = response.data[0].embedding

        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ['documents', 'metadatas']
        }

        if game_name:
            # Assumes 'game_name' field in metadata was populated by ingest.py
            # with a cleaned-up version of the game name.
            query_params["where"] = {"game_name": {"$eq": game_name.title()}} # Case-sensitive match on title-cased name
            # For more flexible matching, consider:
            # query_params["where"] = {"game_name": {"$like": f"%{game_name}%"}} # Substring, might be too broad
            # Or ensure game_name in metadata is normalized during ingestion and here.

        results = collection.query(**query_params)
        
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]

        if not documents:
            return f"No relevant information found in the manuals for '{query}' (game: {game_name if game_name else 'any'}). Try rephrasing or checking the game name."

        formatted_results = []
        for doc, meta in zip(documents, metadatas):
            source_file = meta.get('source_file', 'Unknown Manual')
            game_from_meta = meta.get('game_name', 'Unknown Game')
            chunk_num = meta.get('chunk_num', 'N/A')
            formatted_results.append(f"From '{game_from_meta}' (manual: {source_file}, chunk {chunk_num}):\n{doc}")
        
        return "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        print(f"Error in search_board_game_manuals: {e}")
        return f"Error performing manual search: {str(e)}"


class BGGSearchInput(BaseModel):
    game_name: str = Field(description="The name of the board game to search for on BoardGameGeek (BGG).")
    query_type: str = Field("general_info", description="Type of information: 'general_info', 'rules_faq', 'errata'. Default 'general_info'.")

@tool("search_boardgamegeek", args_schema=BGGSearchInput)
def search_boardgamegeek(game_name: str, query_type: str = "general_info") -> str:
    """
    Searches BoardGameGeek (BGG) for general info, errata, or FAQs for a specific board game.
    """
    print(f"Tool: search_boardgamegeek called for game: '{game_name}', type: '{query_type}'")
    try:
        bgg = get_bgg_client_singleton()
        # The BGG API client might need an exact name, or it might do fuzzy search.
        # For robustness, one might try searching for the game ID first if game_name is ambiguous.
        games = bgg.games(game_name)
        if not games:
            return f"Could not find game '{game_name}' on BoardGameGeek. Please check the spelling."

        game = games[0] # Assume first result is the most relevant
        
        if query_type == "errata" or query_type == "rules_faq":
            return (f"For official errata or community FAQs for '{game.name}' (BGG ID: {game.id}), "
                    f"it's best to check the official forums, files, and wiki sections on its BoardGameGeek page: "
                    f"https://boardgamegeek.com/boardgame/{game.id}")

        # General Info
        description_snippet = game.description
        if description_snippet and len(description_snippet) > 600: # Keep it somewhat brief
            description_snippet = description_snippet[:597] + "..."
        
        info_parts = [
            f"BoardGameGeek Info for {game.name} (ID: {game.id}):",
            f"Description: {description_snippet if description_snippet else 'N/A'}",
            f"Year Published: {game.yearpublished if game.yearpublished else 'N/A'}",
            f"Players: {game.minplayers}-{game.maxplayers}",
            f"Playing Time: {game.minplaytime}-{game.maxplaytime} minutes (approx {game.playingtime})",
            f"Categories: {', '.join(game.categories) if game.categories else 'N/A'}",
            f"Mechanics: {', '.join(game.mechanics) if game.mechanics else 'N/A'}",
        ]
        # game.ranks is a list of dicts, e.g. [{'type': 'subtype', ...}]
        rank_info = "N/A"
        if game.ranks:
            for rank_entry in game.ranks:
                if rank_entry.get('name') == 'boardgame' and rank_entry.get('friendlyname') == 'Board Game Rank':
                    rank_value = rank_entry.get('value', 'Not Ranked')
                    rank_info = f"{rank_value if rank_value != 'Not Ranked' else 'Not Ranked'}"
                    break
        info_parts.append(f"BGG Rank: {rank_info}")
        info_parts.append(f"BGG Link: https://boardgamegeek.com/boardgame/{game.id}")
        return "\n".join(info_parts)

    except Exception as e:
        print(f"Error in search_boardgamegeek: {e}")
        return f"Error searching BGG: {str(e)}"

class AskUserInput(BaseModel):
    clarifying_question: str = Field(description="The question to ask the user to get more specific information or confirm their intent.")

@tool("ask_user_for_clarification", args_schema=AskUserInput)
def ask_user_for_clarification(clarifying_question: str) -> str:
    """
    Use this tool when the user's query is too vague, ambiguous, or needs confirmation to proceed effectively.
    Formulate a specific question to ask the user. The output of this tool will be shown directly to the user,
    and the conversation will pause awaiting their response.
    """
    print(f"Tool: ask_user_for_clarification called with question: '{clarifying_question}'")
    return f"CLARIFICATION_NEEDED: {clarifying_question}"


available_tools = [search_board_game_manuals, search_boardgamegeek, ask_user_for_clarification]