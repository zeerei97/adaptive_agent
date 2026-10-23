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

def input_filter_node(state: AgentState):
    print("--- [Input Filter] Checking the relevance of user input... ---")
    messages = state["messages"]
    last_message = messages[-1]
    
    # Specialized prompt for rapid classification
    filter_prompt = SystemMessage(content=(
        "You are a strict security and domain filter for a specialized Mathematics Agent. "
        "Analyze the user's input. Is it related to mathematics, numbers, calculations, arithmetic, algebra, geometry, or asking for a math tool? "
        "Answer ONLY with 'YES' if it is math-related, or 'NO' if it is completely unrelated (like general knowledge, coding unrelated to math, creative writing, personal chat, etc.)."
    ))
    
    response = llm.invoke([filter_prompt, last_message])
    decision = response.content.strip().upper()
    
    if "NO" in decision:
        # If it is not a match, provide a polite response directly within the state.
        polite_reply = AIMessage(content="Sorry, I am a specialized mathematics agent that can only assist with answering math-related questions or calculations. Please feel free to ask any questions about mathematics!")
        return {"messages": [polite_reply]}
    
    # If YES, allow the user's message to continue being processed by the planner (do not modify the messages)
    return {}

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
    "You are an expert mathematical planning agent. "
    "Your sole objective is to solve the user's validated math problem efficiently. "
    "1. Check the available tools first. If a suitable tool exists, call it immediately. "
    "2. If a required tool does not exist, you MUST explicitly output the phrase 'create a tool' so the system can build it. "
    "3. Never perform calculations manually if a tool can be utilized or created."
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

def filter_router(state: AgentState):
    # Check the last message. If the message is a rejection from a filter, proceed directly to END.
    last_message = state["messages"][-1]
    
    # If the last message contains our rejection statement, direct to END.
    if "Sorry, I am a mathematics agent." in last_message.content:
        return END
        
    return "proceed_to_planner"

def router(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "call_tool"
    
    # Retrieve content securely
    content = last_message.content
    if isinstance(content, list):
        content_str = "".join([item.get("text", "") for item in content if isinstance(item, dict)])
    else:
        content_str = str(content)
        
    content_lower = content_str.lower()
    
    # Expand keywords for tool creation detection
    keywords = ["new tool", "create a tool", "need a tool", "missing tool", "tool needed"]
    if any(kw in content_lower for kw in keywords):
        return "make_tool"
        
    return END

def post_processor_node(state: AgentState):
    print("--- [Post-Processor] Tidying up the final answer & securing the memory... ---")
    messages = state["messages"]
    
    # Take the final raw result (from the executor or planner)
    last_message = messages[-1]
    raw_content = last_message.content
    
    # 1. Format the answer so that it is neat and easy for the user to read.
    formatting_prompt = SystemMessage(content=(
        "You are a helpful mathematical assistant. "
        "Take the raw result/answer and format it into a clean, professional, and clear final response for the user. "
        "Keep it concise and direct."
    ))
    
    format_response = llm.invoke([formatting_prompt, AIMessage(content=str(raw_content))])
    clean_answer = format_response.content
    
    # Create a message with clean
    final_ai_message = AIMessage(content=clean_answer)
    
    # Add the cleaned message to the state so that it can be read by the summarizer function.
    state["messages"].append(final_ai_message)
    
    # 2 & 3. Execute the `summarize_and_clean_state` function for .txt archives and reset the message state.
    cleanup_result = summarize_and_clean_state(state)
    
    # Return a clean message to the state so that the terminal can display the final answer to the user.
    return {
        "messages": cleanup_result["messages"] + [final_ai_message]
    }

# --- 6. BUILDING A GRAPH (LANGGRAPH) ---

workflow = StateGraph(AgentState)

# add nodes
workflow.add_node("input_filter", input_filter_node)
workflow.add_node("planner", planner_node)
workflow.add_node("tool_maker", persistent_tool_maker_node)
workflow.add_node("executor", execution_node)
workflow.add_node("post_processor", post_processor_node)

# Hubungkan entry point ke input_filter
workflow.set_entry_point("input_filter")

# Tambahkan conditional edge dari input_filter
workflow.add_conditional_edges(
    "input_filter",
    filter_router,
    {
        "proceed_to_planner": "planner",
        END: END
    }
)

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
workflow.add_edge("executor", "post_processor")
workflow.add_edge("post_processor", END)

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
            print("\n--- [Sistem] The Agent is processing... ---")
            current_state = app.invoke(current_state)
            
            # Take the final answer from the agent safely
            final_response = current_state["messages"][-1].content
            print(f"\nAI Agent: {final_response}")
            
            # Update tool list in state (jika ada tool baru yang dibuat oleh tool_maker)
            current_state["available_tools"] = refresh_available_tools()
            
        except KeyboardInterrupt:
            print("\nThe program was terminated by the user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
