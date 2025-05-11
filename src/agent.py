import os
import json
from typing import TypedDict, Annotated, Sequence, Literal, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI # Updated import name
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Relative import for tools.py in the same package (src)
from .tools import available_tools

# --- Configuration ---
load_dotenv() # Load .env from root
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_MODEL_NAME = os.getenv("LANGGRAPH_AGENT_MODEL", "gpt-4-turbo-preview")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set.")

# --- Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    # Add any other specific state variables if needed for more complex flows

# --- LLM and Tool Binding ---
llm = ChatOpenAI(model=AGENT_MODEL_NAME, temperature=0, streaming=False) # Streaming handled by Streamlit app if needed
llm_with_tools = llm.bind_tools(tools=available_tools, tool_choice=None) # Allow LLM to choose or respond directly


# --- Agent Nodes ---
def agent_node(state: AgentState) -> dict:
    print(f"---AGENT NODE (History depth: {len(state['messages'])})---")
    # for msg in state['messages']: print(f"  {msg.type}: {msg.content[:100] if msg.content else 'No content'}")
    response = llm_with_tools.invoke(state["messages"])
    # print(f"Agent response: {response}")
    return {"messages": [response]}

tool_node = ToolNode(available_tools)

# --- Conditional Edges ---
def router(state: AgentState) -> Literal["tools", "__end__", "clarification_needed"]:
    print("---ROUTER---")
    last_message = state["messages"][-1]
    # print(f"Router evaluating: {last_message}")

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # print(f"  Tool calls found: {last_message.tool_calls}")
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == "ask_user_for_clarification":
                print("  Router: Clarification tool called.")
                return "clarification_needed"
        print("  Router: Other tools called.")
        return "tools"
    
    print("  Router: No tools called, routing to __end__.")
    return "__end__"


# --- Build the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    router,
    {
        "tools": "tools",
        "clarification_needed": END, # Graph ENDs; Streamlit handles pause & resume
        "__end__": END
    }
)
workflow.add_edge("tools", "agent") # After tools run, go back to agent to process results

agent_graph = workflow.compile()
print("LangGraph agent compiled.")


# --- Helper to run the agent (used by streamlit_app.py) ---
def run_agent_graph_turn(current_conversation_messages: List[BaseMessage]):
    """
    Runs the agent graph for one turn with the given conversation history.
    Yields intermediate agent thoughts/actions or the final response.
    Handles the special "CLARIFICATION_NEEDED" signal.
    `current_conversation_messages` should be the complete history up to this point.
    """
    # The input to the graph is the current state of messages.
    graph_input = {"messages": current_conversation_messages}
    
    # Stream events from the graph
    # Each part is a dictionary with the node name as key and its output as value
    for event_part in agent_graph.stream(graph_input):
        # event_part example: {'agent': {'messages': [AIMessage(..., tool_calls=[...])]}}
        #                     {'tools': {'messages': [ToolMessage(...)]}}
        
        # We are primarily interested in the output of the 'agent' node
        if "agent" in event_part:
            agent_node_output = event_part["agent"] # This is a dict, e.g., {'messages': [AIMessage(...)]}
            # The new message(s) from the agent are in agent_node_output['messages']
            # Typically, it's one AIMessage.
            if agent_node_output and agent_node_output.get("messages"):
                last_ai_message_from_agent_node = agent_node_output["messages"][-1]
                
                if isinstance(last_ai_message_from_agent_node, AIMessage):
                    if last_ai_message_from_agent_node.tool_calls:
                        # Agent wants to call a tool
                        for tc in last_ai_message_from_agent_node.tool_calls:
                            if tc['name'] == 'ask_user_for_clarification':
                                try:
                                    args = json.loads(tc['args']) if isinstance(tc['args'], str) else tc['args']
                                    clarifying_question = args.get("clarifying_question", "I need more details. Could you please clarify?")
                                    yield {
                                        "type": "clarification", 
                                        "content": clarifying_question, 
                                        "tool_call_id": tc['id'],
                                        "ai_message_with_tool_call": last_ai_message_from_agent_node # Pass the whole AIMessage
                                    }
                                    return # Stop this turn, wait for user clarification
                                except json.JSONDecodeError:
                                    print(f"Error decoding tool args: {tc['args']}")
                                    yield {"type": "error", "content": "Agent tried to ask for clarification but failed to format its request."}
                                    return
                        # If other tools were called, the graph will proceed to 'tools' node.
                        # We don't need to yield here for other tools, graph handles it.
                    else:
                        # No tool calls, this is a direct textual response
                        yield {"type": "response", "content": last_ai_message_from_agent_node.content}
                        return # Final response for this turn
        
        # If you wanted to explicitly show tool execution results, you could inspect 'tools' node events
        # elif "tools" in event_part:
        #     tool_node_output = event_part["tools"]
        #     if tool_node_output and tool_node_output.get("messages"):
        #         last_tool_message = tool_node_output["messages"][-1]
        #         yield {"type": "tool_result", "content": f"Tool execution result: {last_tool_message.content[:200]}..."}

    # Fallback: If the stream finishes without a yield (e.g., graph ends due to router logic directly)
    # We need to get the final state to extract the last message.
    # This path might indicate an unexpected graph termination or a need for more explicit yields.
    # print("Warning: Graph stream finished without explicit yield of response or clarification.")
    final_graph_state = agent_graph.invoke(graph_input)
    if final_graph_state and final_graph_state.get("messages"):
        final_message_in_state = final_graph_state["messages"][-1]
        if isinstance(final_message_in_state, AIMessage) and not final_message_in_state.tool_calls:
            yield {"type": "response", "content": final_message_in_state.content}
            return
    
    # If still no clear response
    yield {"type": "response", "content": "I'm not sure how to respond to that. Could you try rephrasing?"}