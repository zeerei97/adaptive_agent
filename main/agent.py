import os
import math
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv
import uuid

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import safer_getattr
from RestrictedPython.PrintCollector import PrintCollector
load_dotenv()

# 1. Definition of State Graph
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    generated_code: str
    tool_approved: bool
    feedback: str
    execution_success: bool
    execution_error: str
    retry_count: int

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    return str(content)

# 2. Sandbox RestrictedPython (Math module preloaded globally)
def run_with_restricted_python(code_text: str) -> str:
    try:
        # Compile code
        byte_code = compile_restricted(code_text, '<string>', 'exec')
    except SyntaxError as e:
        return f"Error SyntaxError: {e}"
    except Exception as e:
        return f"Error kompilasi RestrictedPython: {e}"

    # Prepare a safe environment
    my_globals = safe_globals.copy()
    my_globals['math'] = math  # Preload math modules directly into the global sandbox
    my_globals['_getattr_'] = safer_getattr

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "math":
            return math
        raise ImportError(f"Module '{name}' prohibited by the sandbox security policy.")

    my_globals['_import_'] = safe_import
    restricted_builtins = my_globals['__builtins__'].copy()
    restricted_builtins['__import__'] = safe_import
    my_globals['__builtins__'] = restricted_builtins

    # Container for print() results
    my_globals['_print_'] = PrintCollector

    my_locals = {}

    try:
        exec(byte_code, my_globals, my_locals)
        print_collector = my_locals.get('_print')
        result = print_collector() if print_collector else ""
        return result if result else "Execution successful (no print output)."
    except Exception as e:
        return f"Error during runtime execution: {e}"

# 3. Nodes with Anti-Import Instructions
def code_writer_node(state: AgentState):
    messages = state["messages"]
    feedback = state.get("feedback", "")
    execution_error = state.get("execution_error", "")
    last_message = extract_text(messages[-1].content)
    
    # Set up a fix note outside the main f-string to avoid backslash SyntaxError in Python 3.11
    error_section = ""
    if feedback or execution_error:
        notes = []
        if feedback:
            notes.append(f"- Feedback from humans: {feedback}")
        if execution_error:
            notes.append(f"- Previous execution error in sandbox: {execution_error}\n  (Fix the code so this error doesn't happen again!)")
        
        combined_notes = "\n".join(notes)
        error_section = f"Repair Notes / Error:\n{combined_notes}"

    prompt = f"""
You are a professional Python Code Interpreter.
User request: "{last_message}"
{error_section}

Write complete Python code that solves the above math calculations.
ABSOLUTE Conditions:
1. The `math` module is automatically available in the environment. **DO NOT write `import math`**. Directly use the `math.factorial(...)`, `math.sqrt(...)`, etc. functions.
2. Print the final result of the calculation using the `print(...)` function.
3. Output ONLY valid Python code blocks within the ```python ... ``` markdown without any other explanatory text.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    code_text = extract_text(response.content)
    
    if "```python" in code_text:
        code_text = code_text.split("```python")[1].split("```")[0].strip()
    elif "```" in code_text:
        code_text = code_text.split("```")[1].split("```")[0].strip()
        
    return {
        "generated_code": code_text,
        "execution_error": ""
    }

def human_review_node(state: AgentState):
    code = state["generated_code"]
    human_decision = interrupt({
        "prompt": "The agent has written new Python code. Please review the code:",
        "generated_code": code,
        "instructions": "Type 'approved': True to approve, or 'approved': False with 'feedback' to revise."
    })
    return {
        "tool_approved": human_decision.get("approved", False),
        "feedback": human_decision.get("feedback", "")
    }

def code_executor_node(state: AgentState):
    code = state["generated_code"]
    retry_count = state.get("retry_count", 0)
    
    result = run_with_restricted_python(code)
    is_error = result.startswith("Error") or "Ditolak" in result
    
    if is_error:
        print(f"\n[Sandbox Warning] Error detected, triggers Self-Correction (Attempt to-{retry_count + 1})...")
        return {
            "execution_success": False,
            "execution_error": result,
            "retry_count": retry_count + 1
        }
    else:
        return {
            "execution_success": True,
            "execution_error": "",
            "messages": [AIMessage(content=f"Result of Successful Execution:\n{result}")]
        }

# 4. Routing Logic
def router_after_review(state: AgentState):
    return "execute" if state.get("tool_approved", False) else "rewrite"

def router_after_executor(state: AgentState):
    if not state.get("execution_success", True):
        if state.get("retry_count", 0) < 3:
            return "auto_fix"
        else:
            return "max_retry_exceeded"
    return "end"

# 5. Workflow Compilation
workflow = StateGraph(AgentState)
workflow.add_node("writer", code_writer_node)
workflow.add_node("review", human_review_node)
workflow.add_node("executor", code_executor_node)

workflow.add_edge(START, "writer")
workflow.add_edge("writer", "review")
workflow.add_conditional_edges("review", router_after_review, {
    "execute": "executor",
    "rewrite": "writer"
})
workflow.add_conditional_edges("executor", router_after_executor, {
    "auto_fix": "writer",
    "max_retry_exceeded": END,
    "end": END
})

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ==========================================
# Trial Execution
# ==========================================
def running():
    print("\nHallo, I am math agent to help you")
    
    # Buat thread_id unik menggunakan UUID agar sesi bersih dan tidak konflik
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("\nwhat is your question: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Gunakan variabel 'messages' yang benar (bukan 'query')
        messages = [HumanMessage(content=user_input)]
        
        for event in app.stream({"messages": messages}, config, stream_mode="values"):
            if "messages" in event:
                latest = event["messages"][-1]
                if isinstance(latest, AIMessage) and latest.content:
                    print(f"Agent: {latest.content}")

        state = app.get_state(config)
        if state.next:
            print("\n[Graph Berhenti: Menunggu Peninjauan Manusia]")
            if state.tasks and state.tasks[0].interrupts:
                interrupt_value = state.tasks[0].interrupts[0].value
                print("\n--- KODE PYTHON YANG DIBUAT AGEN ---")
                print(interrupt_value["generated_code"])
                print("-" * 50)
                
                # Menerima input nyata dari terminal Anda
                user_choice = input("Setujui kode ini? (y/n): ").strip().lower()
                is_approved = user_choice == 'y'
                feedback_msg = ""
                
                if not is_approved:
                    feedback_msg = input("Masukkan alasan penolakan/feedback untuk agent: ")
        
                human_response = {
                    "approved": is_approved, 
                    "feedback": feedback_msg
                }
                
                print("\nMelanjutkan eksekusi berdasarkan keputusan Anda...")
                
                for event in app.stream(Command(resume=human_response), config, stream_mode="values"):
                    if "messages" in event:
                        latest = event["messages"][-1]
                        if isinstance(latest, AIMessage) and latest.content:
                            print(f"\n{latest.content}")

# Jalankan program
running()
