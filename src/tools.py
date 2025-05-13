import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from boardgamegeek import BGGClient
from qdrant_client import QdrantClient, models # Qdrant imports
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# --- Environment and Clients ---
load_dotenv() # Load .env from root
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "board_game_manuals")
EMBEDDING_MODEL = "text-embedding-ada-002"

if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY not set.")
if not QDRANT_URL: raise ValueError("QDRANT_URL not set.")

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except TypeError:
    raise ImportError("OpenAI library version issue. Ensure openai>=1.0.0.")

_qdrant_db_client = None
_bgg_client = None

def get_qdrant_client_singleton():
    global _qdrant_db_client
    if _qdrant_db_client is None:
        try:
            _qdrant_db_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
            _qdrant_db_client.get_collections() # Test connection quickly
        except Exception as e:
            _qdrant_db_client = None
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")
    return _qdrant_db_client

def get_bgg_client_singleton():
    # (Same as before)
    global _bgg_client
    if _bgg_client is None: _bgg_client = BGGClient()
    return _bgg_client

# --- Tool Definitions ---

class QdrantSearchInput(BaseModel):
    query: str = Field(description="The question or topic to search for in the board game manuals.")
    game_name: str = Field(None, description="Optional: The specific game name (e.g., 'Catan', 'Ticket To Ride') to filter search results for.")
    top_k: int = Field(3, description="Number of relevant document chunks to retrieve.")

@tool("search_board_game_manuals", args_schema=QdrantSearchInput)
def search_board_game_manuals(query: str, game_name: str = None, top_k: int = 3) -> str:
    """
    Searches the vector database of ingested board game manuals for information relevant to the query.
    Use this to answer specific rule questions. If a specific game is mentioned, provide the cleaned game_name (e.g., 'Catan', 'Ticket To Ride').
    """
    print(f"Tool: search_board_game_manuals called with query: '{query}', game: '{game_name}', top_k: {top_k}")
    try:
        qdrant_client = get_qdrant_client_singleton()

        # 1. Embed the query
        response = openai_client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
        query_embedding = response.data[0].embedding

        # 2. Prepare Qdrant search filter (if game_name provided)
        search_filter = None
        if game_name:
            # Assumes 'game_name' is stored in payload.metadata.game_name
            # Needs exact match with the cleaned game name stored during ingestion
            cleaned_filter_game_name = game_name.strip().title() # Ensure consistent casing
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.game_name", # Access nested field
                        match=models.MatchValue(value=cleaned_filter_game_name)
                    )
                ]
            )
            print(f"  Applying Qdrant filter for game: {cleaned_filter_game_name}")

        # 3. Perform the search
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=search_filter, # Pass filter if created
            limit=top_k,
            with_payload=True # Ensure we get the payload back
        )
        # search_result is a list of ScoredPoint objects

        if not search_result:
            return f"No relevant information found in the manuals for '{query}' (game: {game_name if game_name else 'any'}). Try rephrasing or check the game name matches the supported list."

        # 4. Format the results
        formatted_results = []
        for hit in search_result:
            payload = hit.payload # Payload is a dict
            metadata = payload.get("metadata", {}) # Access nested metadata dict
            text_chunk = payload.get("text", "Missing text in payload")
            
            source_file = metadata.get('source_file', 'Unknown Manual')
            game_from_meta = metadata.get('game_name', 'Unknown Game')
            chunk_num = metadata.get('chunk_num', 'N/A')
            score = hit.score # Similarity score

            formatted_results.append(
                f"From '{game_from_meta}' (manual: {source_file}, chunk {chunk_num}, score: {score:.4f}):\n{text_chunk}"
            )
        
        return "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        print(f"Error in search_board_game_manuals (Qdrant): {e}")
        # Provide a more informative error message back to the agent/LLM
        return f"Error searching manuals: {type(e).__name__} - {str(e)}. Please check connection or query."


# BGG Search tool (same as before)
class BGGSearchInput(BaseModel):
    game_name: str = Field(description="The name of the board game to search for on BoardGameGeek (BGG).")
    query_type: str = Field("general_info", description="Type of information: 'general_info', 'rules_faq', 'errata'. Default 'general_info'.")

@tool("search_boardgamegeek", args_schema=BGGSearchInput)
def search_boardgamegeek(game_name: str, query_type: str = "general_info") -> str:
    """Search for a board game on BoardGameGeek by name."""
    print(f"Tool: search_boardgamegeek called for game: '{game_name}', type: '{query_type}'")
    try:
        bgg = get_bgg_client_singleton(); games = bgg.games(game_name)
        if not games: return f"Could not find game '{game_name}' on BGG."
        game = games[0]
        if query_type == "errata" or query_type == "rules_faq":
            return (f"For errata/FAQs for '{game.name}' (BGG ID: {game.id}), "
                    f"check the forums/files/wiki on its BGG page: "
                    f"https://boardgamegeek.com/boardgame/{game.id}")
        desc = game.description[:600] + "..." if game.description and len(game.description) > 600 else game.description or 'N/A'
        rank = "N/A"
        if game.ranks:
            for r in game.ranks:
                if r.get('name') == 'boardgame': rank = r.get('value', 'Not Ranked'); break
        info = [f"BGG Info for {game.name} (ID: {game.id}):", f"Desc: {desc}", f"Year: {game.yearpublished or 'N/A'}",
                f"Players: {game.minplayers}-{game.maxplayers}", f"Time: {game.minplaytime}-{game.maxplaytime} min",
                f"Categories: {', '.join(game.categories) if game.categories else 'N/A'}",
                f"Mechanics: {', '.join(game.mechanics) if game.mechanics else 'N/A'}",
                f"Rank: {rank}", f"Link: https://boardgamegeek.com/boardgame/{game.id}"]
        return "\n".join(info)
    except Exception as e: print(f"Error in search_boardgamegeek: {e}"); return f"Error searching BGG: {str(e)}"


# Clarification tool (updated with docstring)
class AskUserInput(BaseModel):
    clarifying_question: str = Field(description="The question to ask the user to get more specific information or confirm their intent.")

@tool("ask_user_for_clarification", args_schema=AskUserInput)
def ask_user_for_clarification(clarifying_question: str) -> str:
    """Ask the user for clarification on a question or instruction."""
    print(f"Tool: ask_user_for_clarification called with question: '{clarifying_question}'")
    return f"CLARIFICATION_NEEDED: {clarifying_question}"

# --- Available Tools List ---
available_tools = [search_board_game_manuals, search_boardgamegeek, ask_user_for_clarification]