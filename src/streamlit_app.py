import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Absolute import for agent.py
from src.agent import run_agent_graph_turn

# --- Configuration ---
load_dotenv()
# Use QDRANT_COLLECTION_NAME from env var for display
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "board_game_manuals")
# Path inside Docker container (assuming data/ is copied to /app/data)
SUPPORTED_GAMES_FILE = os.path.join("data", "supported_games.txt")

# --- Helper Functions ---
@st.cache_data
def load_supported_games():
    # (Identical to previous Docker version)
    try:
        if os.path.exists(SUPPORTED_GAMES_FILE):
            with open(SUPPORTED_GAMES_FILE, "r") as f: games = [line.strip() for line in f if line.strip()]
            if not games: return ["No games processed yet. Run `ingest.py` locally and commit `data/supported_games.txt`."]
            return games
        else: return ["`data/supported_games.txt` not found. Ensure it's committed and Dockerfile copies `data/`."]
    except Exception as e: print(f"Error loading games file: {e}"); return [f"Error loading list: {e}"]

# --- Streamlit App UI ---
st.set_page_config(page_title="Board Game Manual Agent", layout="wide")
st.title("🎲 Board Game Manual Agent")
# Updated caption to mention Qdrant and correct collection name env var
st.caption(f"Ask rule questions or for game info! Powered by LangGraph, OpenAI & Qdrant Cloud. Manuals Collection: `{QDRANT_COLLECTION_NAME}`")

# Sidebar
with st.sidebar:
    # (Identical to previous version, including Clear Chat button)
    st.header("📚 Supported Game Manuals")
    supported_games = load_supported_games()
    for g in supported_games:
        st.markdown(f"- {g}")
    st.markdown("---")
    st.info("Searches local manuals (via Qdrant), BoardGameGeek, or asks for clarification.")
    if st.button("Clear Chat History & Reset Agent"):
        st.session_state.display_messages = [{"role": "assistant", "content": "Hi! How can I help?"}]
        st.session_state.langgraph_conversation_history = []
        st.session_state.waiting_for_clarification = False; st.session_state.clarification_context = {}
        st.rerun()


# Initialize session state variables
# (Identical to previous version)
if "display_messages" not in st.session_state: st.session_state.display_messages = [{"role": "assistant", "content": "Hi! How can I help?"}]
if "langgraph_conversation_history" not in st.session_state: st.session_state.langgraph_conversation_history = []
if "waiting_for_clarification" not in st.session_state: st.session_state.waiting_for_clarification = False
if "clarification_context" not in st.session_state: st.session_state.clarification_context = {}

# Display chat messages
# (Identical to previous version)
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# Handle user input
user_prompt = st.chat_input("Ask about rules, BGG, or just chat...")

if user_prompt:
    # (Input handling logic identical to previous version)
    st.session_state.display_messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"): st.markdown(user_prompt)

    current_turn_input_messages = list(st.session_state.langgraph_conversation_history)

    if st.session_state.waiting_for_clarification:
        ai_message_that_asked = st.session_state.clarification_context.get("ai_message_with_tool_call")
        tool_call_id = st.session_state.clarification_context.get("tool_call_id")
        if ai_message_that_asked and tool_call_id:
             # History should already contain the AIMessage that asked. Now add user's clarification ToolMessage.
            current_turn_input_messages.append(ToolMessage(content=user_prompt, tool_call_id=tool_call_id))
        else:
            st.error("Error: Missing context for clarification. Please try again."); st.stop()
        st.session_state.waiting_for_clarification = False; st.session_state.clarification_context = {}
    else:
        current_turn_input_messages.append(HumanMessage(content=user_prompt))

    with st.chat_message("assistant"):
        message_placeholder = st.empty(); message_placeholder.markdown("Thinking...⚙️")
        final_response_content = None
        try:
            # Core agent interaction loop (Identical to previous version)
            for event in run_agent_graph_turn(current_turn_input_messages):
                if event["type"] == "clarification":
                    clarifying_question = event["content"]
                    st.session_state.display_messages.append({"role": "assistant", "content": clarifying_question})
                    message_placeholder.markdown(clarifying_question)
                    st.session_state.waiting_for_clarification = True
                    st.session_state.clarification_context = { # Store context
                        "tool_call_id": event["tool_call_id"],
                        "ai_message_with_tool_call": event["ai_message_with_tool_call"]
                    }
                    # Update history up to the point of the AI asking
                    st.session_state.langgraph_conversation_history = current_turn_input_messages + [event["ai_message_with_tool_call"]]
                    final_response_content = None; break
                elif event["type"] == "response":
                    final_response_content = event["content"]
                    message_placeholder.markdown(final_response_content)
                    # Update history with the final AI response for this turn
                    st.session_state.langgraph_conversation_history = current_turn_input_messages + [AIMessage(content=final_response_content)]
                    break
                elif event["type"] == "error":
                    final_response_content = event["content"]; st.error(final_response_content); message_placeholder.markdown(f"Error: {final_response_content}")
                    st.session_state.langgraph_conversation_history = current_turn_input_messages + [AIMessage(content=f"Agent Error: {final_response_content}")]
                    break

            # UI updates after loop (Identical to previous version)
            if final_response_content is not None:
                st.session_state.display_messages.append({"role": "assistant", "content": final_response_content})
                message_placeholder.markdown(final_response_content)
            elif not st.session_state.waiting_for_clarification:
                err_msg = "Sorry, I couldn't process that."; st.session_state.display_messages.append({"role": "assistant", "content": err_msg}); message_placeholder.markdown(err_msg)
                st.session_state.langgraph_conversation_history = current_turn_input_messages + [AIMessage(content=err_msg)]

        except Exception as e:
            # Error handling (Identical to previous version)
            st.error(f"An application error occurred: {e}"); import traceback; st.error(traceback.format_exc())
            error_msg = f"Sorry, an unexpected error occurred: {str(e)[:100]}..."
            st.session_state.display_messages.append({"role": "assistant", "content": error_msg})
            st.session_state.waiting_for_clarification = False; st.session_state.clarification_context = {}
            st.session_state.langgraph_conversation_history = current_turn_input_messages + [AIMessage(content=f"System Error: {e}")]

# Debug Expander (Identical)
# with st.expander("View LangGraph History"): st.json(...)