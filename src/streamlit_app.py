import streamlit as st
import os
import json # For handling tool call arguments if they are strings
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Relative import for agent.py in the same package (src)
from .agent import run_agent_graph_turn

# --- Configuration ---
load_dotenv() # Load .env from root project directory
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "board_game_manuals")
# Path inside Docker container, assuming Dockerfile copies `data` to `/app/data`
# and WORKDIR is /app
SUPPORTED_GAMES_FILE = os.path.join("data", "supported_games.txt")

# --- Helper Functions ---
@st.cache_data
def load_supported_games():
    # This path is relative to WORKDIR /app inside the Docker container
    try:
        if os.path.exists(SUPPORTED_GAMES_FILE):
            # print(f"Attempting to load from: {os.path.abspath(SUPPORTED_GAMES_FILE)}")
            with open(SUPPORTED_GAMES_FILE, "r") as f:
                games = [line.strip() for line in f if line.strip()]
            if not games: return ["No games processed yet. Run `ingest.py` locally and commit `data/supported_games.txt`."]
            return games
        else:
            # print(f"File not found: {os.path.abspath(SUPPORTED_GAMES_FILE)}")
            # For debugging in HF Spaces logs:
            # print(f"Current dir: {os.getcwd()}")
            # print(f"/app contents: {os.listdir('/app') if os.path.exists('/app') else 'N/A'}")
            # print(f"/app/data contents: {os.listdir('/app/data') if os.path.exists('/app/data') else 'N/A'}")
            return ["`data/supported_games.txt` not found. Ensure it's committed and Dockerfile copies `data/`."]
    except Exception as e:
        print(f"Error loading supported games file from '{SUPPORTED_GAMES_FILE}': {e}")
        return [f"Error loading list of supported games: {e}"]

# --- Streamlit App UI ---
st.set_page_config(page_title="Board Game Manual Agent", layout="wide")
st.title("🎲 Board Game Manual Agent")
st.caption(f"Ask rule questions or for game info! Powered by LangGraph & OpenAI. Manuals Collection: `{DEFAULT_COLLECTION_NAME}`")

# Sidebar
with st.sidebar:
    st.header("📚 Supported Game Manuals")
    supported_games = load_supported_games()
    for game_name in supported_games:
        st.markdown(f"- {game_name}")
    st.markdown("---")
    st.info("This agent can search local manuals, query BoardGameGeek, or ask for clarification.")
    if st.button("Clear Chat History & Reset Agent"):
        st.session_state.display_messages = [{"role": "assistant", "content": "Hi! How can I help you with board game rules or info today?"}]
        st.session_state.langgraph_conversation_history = []
        st.session_state.waiting_for_clarification = False
        st.session_state.clarification_context = {}
        st.rerun()


# Initialize chat history for display and for LangGraph
if "display_messages" not in st.session_state: # For Streamlit UI
    st.session_state.display_messages = [{"role": "assistant", "content": "Hi! How can I help you with board game rules or info today?"}]
if "langgraph_conversation_history" not in st.session_state: # For LangGraph (list of BaseMessage)
    st.session_state.langgraph_conversation_history = []
if "waiting_for_clarification" not in st.session_state: # Flag
    st.session_state.waiting_for_clarification = False
if "clarification_context" not in st.session_state: # To store info needed to respond to clarification
    st.session_state.clarification_context = {}


# Display chat messages from display_messages
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
user_prompt = st.chat_input("Ask about rules, BGG, or just chat...")

if user_prompt:
    # Add user message to Streamlit display
    st.session_state.display_messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"): # Display immediately
        st.markdown(user_prompt)

    # Prepare messages for the agent for this turn
    current_turn_input_messages = list(st.session_state.langgraph_conversation_history) # Start with existing history

    if st.session_state.waiting_for_clarification:
        # User's input is the clarification
        ai_message_that_asked = st.session_state.clarification_context.get("ai_message_with_tool_call")
        tool_call_id = st.session_state.clarification_context.get("tool_call_id")

        if ai_message_that_asked and tool_call_id:
            # Add the original AIMessage (that called the clarification tool) to history if not already there.
            # The design is that langgraph_conversation_history should already have it.
            # Then, add the ToolMessage (user's answer).
            current_turn_input_messages.append(
                ToolMessage(content=user_prompt, tool_call_id=tool_call_id)
            )
        else:
            st.error("Error: Missing context for clarification. Please try your original query again.")
            st.session_state.waiting_for_clarification = False # Reset
            st.stop() # Stop further processing for this input

        st.session_state.waiting_for_clarification = False
        st.session_state.clarification_context = {}
    else:
        # This is a new query, add it as a HumanMessage for the agent
        current_turn_input_messages.append(HumanMessage(content=user_prompt))

    # Display thinking message for assistant
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...⚙️")
        
        final_response_content = None
        try:
            # Stream agent events for this turn
            for event in run_agent_graph_turn(current_turn_input_messages):
                if event["type"] == "clarification":
                    clarifying_question = event["content"]
                    st.session_state.display_messages.append({"role": "assistant", "content": clarifying_question})
                    message_placeholder.markdown(clarifying_question)
                    
                    # Store context needed for when user provides clarification
                    st.session_state.waiting_for_clarification = True
                    st.session_state.clarification_context = {
                        "tool_call_id": event["tool_call_id"],
                        "ai_message_with_tool_call": event["ai_message_with_tool_call"] # This is the AIMessage object
                    }
                    # Update langgraph_conversation_history up to the point of AIMessage asking for clarification
                    st.session_state.langgraph_conversation_history = current_turn_input_messages + [event["ai_message_with_tool_call"]]
                    final_response_content = None # No final textual response yet
                    break # Stop agent processing for this turn, wait for user's input

                elif event["type"] == "response":
                    final_response_content = event["content"]
                    message_placeholder.markdown(final_response_content) # Display final response
                    # Update langgraph_conversation_history with this completed turn
                    st.session_state.langgraph_conversation_history = current_turn_input_messages + [AIMessage(content=final_response_content)]
                    break # Final response received for this turn
                
                elif event["type"] == "error": # Handle errors from agent/tools
                    final_response_content = event["content"]
                    st.error(final_response_content)
                    message_placeholder.markdown(f"Error: {final_response_content}")
                    # Decide how to update history on error, maybe just add error as AI message
                    st.session_state.langgraph_conversation_history = current_turn_input_messages + [AIMessage(content=f"Agent Error: {final_response_content}")]
                    break


            if final_response_content is not None: # If a response was formed (not just clarification)
                st.session_state.display_messages.append({"role": "assistant", "content": final_response_content})
                # Ensure placeholder is updated if streaming was used or loop exited early
                message_placeholder.markdown(final_response_content)
            elif not st.session_state.waiting_for_clarification: # No response and not waiting means something else
                err_msg = "Sorry, I couldn't process that. Please try again or rephrase."
                st.session_state.display_messages.append({"role": "assistant", "content": err_msg})
                message_placeholder.markdown(err_msg)
                st.session_state.langgraph_conversation_history = current_turn_input_messages + [AIMessage(content=err_msg)]


        except Exception as e:
            st.error(f"An application error occurred: {e}")
            import traceback
            st.error(traceback.format_exc())
            error_message_for_user = f"Sorry, an unexpected error occurred: {str(e)[:100]}..."
            st.session_state.display_messages.append({"role": "assistant", "content": error_message_for_user})
            # Reset clarification state on hard error
            st.session_state.waiting_for_clarification = False
            st.session_state.clarification_context = {}
            # Add error to history so agent is aware if it continues
            st.session_state.langgraph_conversation_history = current_turn_input_messages + [AIMessage(content=f"System Error: {e}")]

# For debugging the conversation state
# with st.expander("View LangGraph Conversation History (for debugging)"):
#    if st.session_state.langgraph_conversation_history:
#        # Use .dict() for Pydantic models like BaseMessage, or .model_dump() for v2
#        st.json([msg.dict() if hasattr(msg, 'dict') else str(msg) for msg in st.session_state.langgraph_conversation_history])