import operator
import inspect
import os
import importlib
import tools
import ast
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Union, List, Callable, Any

# if you wanna use open ai
# from langchain_openai import ChatOpenAI
# from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai._function_utils import convert_to_genai_function_declarations

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, RemoveMessage
from langchain_core.tools import tool, BaseTool
from langgraph.graph import StateGraph, END

load_dotenv()

# --- 1. LLM SETTING MODEL ---
# I want to minimalize hallucinations - temperature = 0 makes the model output more deterministic
llm = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite", temperature=0)

# --- 2. STATE DEFINITION ---
class AgentState(TypedDict):
    # Savinng All Message
    messages: Annotated[List[BaseMessage], operator.add]
    # Saving a dynamic list of equipment
    available_tools: List[Callable]

def summarize_and_clean_state(state: AgentState):
    print("--- [Summarizer] Summarizing the completed text and clearing the state... ---")
    messages = state["messages"]
    
    if not messages:
        return state

    # 1. Ask the LLM to generate a summary of the current task conversation.
    summary_prompt = [
        SystemMessage(content="Provide a concise and compact summary of the mathematical conversation/task below, covering the user's question, the tool(s) created or used, and the final result."),
    ] + messages
    
    summary_response = llm.invoke(summary_prompt)
    task_summary = summary_response.content
    
    # 2. Save the summary permanently to a .txt file
    with open("task_history.txt", "a", encoding="utf-8") as f:
        f.write(f"\n--- Task Summary ---\n{task_summary}\n")
    print("[Summarizer] Summary successfully saved to 'task_history.txt'")
    
    # 3. Create a command to delete the entire history of old messages in the LangGraph state
    # Use RemoveMessage for each existing message ID
    delete_messages = [RemoveMessage(id=msg.id) for msg in messages if hasattr(msg, "id") and msg.id]
    
    # 4. (Optional) Insert the aforementioned summary as a new SystemMessage so that the agent retains global context.
    summary_memory_message = SystemMessage(content=f"Previous task history: {task_summary}")
    
    return {
        "messages": delete_messages + [summary_memory_message]
    }

# --- 3. DEFINITION OF ALL NODE AGENTS ---

def planner_node(state: AgentState):
    """
    Brain Nodes: Analyze requests, select tools, or decide to create new tools.
    """
    print("--- [Planner] Thinking... ---")
    messages = state["messages"]
    tools = state["available_tools"]
    
    # Bind the currently available tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt = SystemMessage(content=(
        "You are an advanced mathematical AI agent."
        "You have access to a set of dynamic tools."
        "Analyze the user's mathematical request."
        "If a suitable tool exists, call it."
        "If you do NOT have a tool to solve the specific operation (e.g., subtraction, multiplication),"
        "explicitly state that you need a new tool made, describing what the tool should do."
    ))
    
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}

# --- 4. AST GUARDRAIL (SAFETY BARRIER) ---
FORBIDDEN_MODULES = {"os", "sys", "subprocess", "shutil", "socket", "requests", "urllib", "pathlib"}
FORBIDDEN_BUILTINS = {"eval", "exec", "open", "__import__", "compile"}

def validate_code_safety(code_str: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in FORBIDDEN_MODULES:
                    return False, f"import of modules is prohibited: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in FORBIDDEN_MODULES:
                return False, f"Imports prohibited from: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
                return False, f"Calling functions is prohibited: {node.func.id}()"
    return True, "Safe"

def refresh_available_tools() -> list:
    """
    Automatically scan the tools.py file to retrieve
    all functions that are designated as tools.
    """
    # 1. Reload the module to load the latest code from the physical file
    importlib.reload(tools)
    
    refreshed_tools = []
    
    # 2. Import all members (functions/variables) from the tools.py module.
    for name, obj in inspect.getmembers(tools, inspect.isfunction):
        # Check whether the object is a LangChain tool
        # (it typically has a 'tool_call' attribute or is an instance of BaseTool)
        if hasattr(obj, "name") and hasattr(obj, "invoke"):
            refreshed_tools.append(obj)
            
    return refreshed_tools

# --- 5. NODE TOOL MAKER WITH PERMANENT STORAGE ---
def persistent_tool_maker_node(state: AgentState):
    print("--- [Tool Maker] Creating a New Tool and Saving It to a File... ---")
    messages = state["messages"]
    last_message = messages[-1]
    
    # 1. Ask an LLM to generate Python function code that includes the @tool decorator.
    maker_prompt = SystemMessage(content=(
        "You are an expert Python developer."
        "Write a clean Python function using the @tool decorator from langchain_core.tools."
        "Include clear type hints and docstrings."
        "ONLY output valid Python code without markdown code blocks."
    ))
    
    response = llm.invoke([maker_prompt, HumanMessage(content=f"Create a tool for: {last_message.content}")])
    generated_code = response.content.strip()
    
    # Clean up any markdown if the LLM inadvertently includes it (e.g., ` ```python ... ``` `)
    if generated_code.startswith("```"):
        generated_code = generated_code.split("```")[1]
        if generated_code.startswith("python"):
            generated_code = generated_code[6:].strip()
            
    # 2. AST Security Filter / Validation
    is_safe, msg = validate_code_safety(generated_code)
    if not is_safe:
        print(f"[SECURITY REJECTED] Code rejected for security reasons: {msg}")
        return {"messages": [AIMessage(content=f"Tool creation was blocked by the security system: {msg}")]}
    
    # 3. Save permanently to the file 'tools.py'
    try:
        with open("tools.py", "a", encoding="utf-8") as f:
            f.write(f"\n\n{generated_code}\n")
        print("[Tool Maker] Successfully added a new tool to the file 'tools.py'!")
        
        # Retrieve all tool functions from the updated tools.py file.
        updated_tools_list = refresh_available_tools
        
        return {"available_tools": updated_tools_list}
        
    except Exception as e:
        print(f"Failed to save to file: {e}")
        return {"messages": [AIMessage(content=f"Failed to save tool to file: {e}")]}

def execution_node(state: AgentState):
    """
    Executor Node: Executes the tool calls requested by the Planner.
    """
    print("--- [Eksekutor] Running Tool... ---")
    messages = state["messages"]
    last_message = messages[-1]
    tools = state["available_tools"]
    
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    
    # Find tool function by name
    chosen_tool = next((t for t in tools if t.name == tool_name), None)
    
    if chosen_tool:
        result = chosen_tool.invoke(tool_args)
        print(f"[Eksekutor] Result {tool_name}({tool_args}): {result}")
        # Return as an AIMessage with a tool_call_id
        return {"messages": [AIMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_name)]}
    else:
        return {"messages": [AIMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call["id"])]}

# --- 6. DEFINITION OF DIRECTION LOGIC (ROUTER) ---

def router(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # 1. If the Planner calls a tool
    if last_message.tool_calls:
        return "call_tool"
    
    # Retrieve content safely (can be a string or a list)
    content = last_message.content
    if isinstance(content, list):
        # If it is a list (e.g., a content block), combine the text into a string.
        content_str = "".join([item.get("text", "") for item in content if isinstance(item, dict)])
    else:
        content_str = str(content)
    
    # 2. If the Planner explicitly wishes to create a new tool within their text
    if "new tool" in content_str.lower() or "create a tool" in content_str.lower():
        return "make_tool"
    
    # 3. If no further action is taken, end the conversation
    return END

# --- 6. BUILDING A GRAPH (LANGGRAPH) ---

workflow = StateGraph(AgentState)

# add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("tool_maker", persistent_tool_maker_node)
workflow.add_node("executor", execution_node)

# Define Control Flow (Edges and Routers)
workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    router,
    {
        "call_tool": "executor",
        "make_tool": "tool_maker",
        END: END
    }
)

# After creating the tool, return to Planner to re-evaluate
workflow.add_edge("tool_maker", "planner")
# After running the tool, return to Planner to summarize the answers
workflow.add_edge("executor", "planner")

# Compile Graph
app = workflow.compile()

# --- 7. STARTING THE AGENT ---

if __name__ == "__main__":
    print("\n=== GETTING STARTED WITH AN ADAPTIVE AND INTERACTIVE MATHEMATICS AGENT ===\n")
    print("Type a math question (example: 'What is 10 plus 5' or '15 minus 7').")
    print("Type 'exit' or 'done' to stop.\n")
    
    # 1. Initialize the initial state by loading all tools from the tools.py file automatically
    current_state = {
        "messages": [],
        "available_tools": refresh_available_tools()
    }
    
    # 2. Interactive loop to accept user input
    while True:
        try:
            user_input = input("\nUser: ").strip()
            
            if user_input.lower() in ["exit", "done"]:
                print("Exit the program. See you!")
                break
                
            if not user_input:
                continue
                
            # Add new messages from users to the messages history in the state
            current_state["messages"].append(HumanMessage(content=user_input))
            
            # Run the graph in its entirety. 
            # The agent (Planner) will automatically decide whether to use the old tool, create a new tool, or answer directly.
            print("\n--- [Sistem] The Agent is processing... ---")
            current_state = app.invoke(current_state)
            
            # Take the final answer from the agent
            final_response = current_state["messages"][-1].content
            print(f"\nAI Agent: {final_response}")
            
            # Update memory and tool list
            current_state["messages"] = summarize_and_clean_state(current_state)["messages"]
            current_state["available_tools"] = refresh_available_tools()
            
        except KeyboardInterrupt:
            print("\nThe program was terminated by the user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")