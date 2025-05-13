# src/agent.py
import os
import json
from typing import TypedDict, Annotated, Sequence, Literal, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Relative import for tools.py in the same package (src)
from .tools import available_tools

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_MODEL_NAME = os.getenv("LANGGRAPH_AGENT_MODEL", "gpt-4-turbo-preview")

if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY not set.")

# --- Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]

# --- LLM and Tool Binding ---
llm = ChatOpenAI(model=AGENT_MODEL_NAME, temperature=0, streaming=False)
llm_with_tools = llm.bind_tools(tools=available_tools, tool_choice=None)

# --- Agent Nodes ---
def agent_node(state: AgentState) -> dict:
    # (Identical to previous version)
    print(f"---AGENT NODE (History depth: {len(state['messages'])})---")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(available_tools)

# --- Conditional Edges ---
def router(state: AgentState) -> Literal["tools", "__end__", "clarification_needed"]:
    # (Identical to previous version)
    print("---ROUTER---")
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == "ask_user_for_clarification": return "clarification_needed"
        return "tools"
    return "__end__"

# --- Build the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node); workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", router, {"tools": "tools", "clarification_needed": END, "__end__": END})
workflow.add_edge("tools", "agent")
agent_graph = workflow.compile()
print("LangGraph agent compiled.")

# --- Helper to run the agent (used by streamlit_app.py) ---
def run_agent_graph_turn(current_conversation_messages: List[BaseMessage]):
    graph_input = {"messages": current_conversation_messages}
    last_ai_message_with_tool_call = None # To capture the message asking for clarification
    for event_part in agent_graph.stream(graph_input):
        if "agent" in event_part:
            agent_node_output = event_part["agent"]
            if agent_node_output and agent_node_output.get("messages"):
                last_ai_msg = agent_node_output["messages"][-1]
                if isinstance(last_ai_msg, AIMessage):
                    if last_ai_msg.tool_calls:
                        last_ai_message_with_tool_call = last_ai_msg # Store it
                        for tc in last_ai_msg.tool_calls:
                            if tc['name'] == 'ask_user_for_clarification':
                                try:
                                    args = json.loads(tc['args']) if isinstance(tc['args'], str) else tc['args']
                                    question = args.get("clarifying_question", "Could you clarify?")
                                    yield {
                                        "type": "clarification", 
                                        "content": question, 
                                        "tool_call_id": tc['id'],
                                        "ai_message_with_tool_call": last_ai_message_with_tool_call # Pass the AIMessage
                                    }
                                    return 
                                except Exception as e:
                                    print(f"Error parsing clarification args: {e}"); yield {"type": "error", "content": "Agent clarification request failed."}; return
                        # If other tools called, graph continues to tool_node
                    else: # Direct response
                        yield {"type": "response", "content": last_ai_msg.content}
                        return
        # elif "tools" in event_part: # Optional: yield tool results for visibility
            # print(f"Tool Node Output: {event_part['tools']}")

    # Fallback if stream ends unexpectedly
    final_graph_state = agent_graph.invoke(graph_input)
    if final_graph_state and final_graph_state.get("messages"):
        final_msg = final_graph_state["messages"][-1]
        if isinstance(final_msg, AIMessage) and not final_msg.tool_calls:
            yield {"type": "response", "content": final_msg.content}
            return
    yield {"type": "response", "content": "Sorry, I couldn't determine the next step."}