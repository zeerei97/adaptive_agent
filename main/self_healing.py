import os
import math
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import safer_getattr

# 1. Definisi State Graph yang Diperbarui (Menambah Error & Retry Tracking)
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    generated_code: str
    tool_approved: bool
    feedback: str
    execution_success: bool
    execution_error: str
    retry_count: int

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

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

# 2. Sandbox RestrictedPython
def run_with_restricted_python(code_text: str) -> str:
    try:
        byte_code = compile_restricted(code_text, '<string>', 'exec')
    except SyntaxError as e:
        return f"Error SyntaxError: {e}"
    except Exception as e:
        return f"Error kompilasi RestrictedPython: {e}"

    my_globals = safe_globals.copy()
    my_globals['math'] = math
    my_globals['_getattr_'] = safer_getattr

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        allowed_modules = {"math"}
        if name in allowed_modules:
            return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f"Modul '{name}' dilarang oleh kebijakan keamanan sandbox.")

    my_globals['__import__'] = safe_import
    my_globals['_import_'] = safe_import

    printed_values = []
    def safe_print(*args, **kwargs):
        output = " ".join(map(str, args))
        printed_values.append(output)

    my_globals['_print'] = safe_print
    my_globals['_print_'] = safe_print
    my_globals['printed'] = printed_values

    my_locals = {}

    try:
        exec(byte_code, my_globals, my_locals)
        result = "\n".join(printed_values)
        return result if result else "Eksekusi berhasil (tanpa output print)."
    except Exception as e:
        return f"Error saat eksekusi runtime: {e}"

# 3. Nodes
def code_writer_node(state: AgentState):
    messages = state["messages"]
    feedback = state.get("feedback", "")
    execution_error = state.get("execution_error", "")
    last_message = extract_text(messages[-1].content)
    
    # Siapkan catatan perbaikan di luar f-string utama untuk menghindari SyntaxError backslash di Python 3.11
    error_section = ""
    if feedback or execution_error:
        notes = []
        if feedback:
            notes.append(f"- Feedback dari manusia: {feedback}")
        if execution_error:
            notes.append(f"- Error eksekusi sebelumnya di sandbox: {execution_error}\n  (Perbaiki kodenya agar error ini tidak terjadi lagi!)")
        
        combined_notes = "\n".join(notes)
        error_section = f"Catatan Perbaikan / Error:\n{combined_notes}"

    prompt = f"""
Kamu adalah seorang Python Code Interpreter profesional.
Permintaan pengguna: "{last_message}"
{error_section}

Tuliskan kode Python lengkap yang menyelesaikan perhitungan matematika di atas.
Ketentuan MUTLAK:
1. Modul `math` sudah tersedia secara otomatis di lingkungan. **JANGAN menulis `import math`**. Langsung gunakan fungsi `math.factorial(...)`, `math.sqrt(...)`, dll.
2. Cetak hasil akhir perhitungan menggunakan fungsi `print(...)`.
3. Keluarkan HANYA blok kode Python valid di dalam markdown ```python ... ``` tanpa teks penjelasan lain.
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
        "prompt": "Agen telah menulis kode Python baru. Mohon tinjau kodenya:",
        "generated_code": code,
        "instructions": "Ketik 'approved': True untuk menyetujui, atau 'approved': False dengan 'feedback' untuk merevisi."
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
        print(f"\n[Sandbox Warning] Terdeteksi Error, memicu Self-Correction (Percobaan ke-{retry_count + 1})...")
        return {
            "execution_success": False,
            "execution_error": result,
            "retry_count": retry_count + 1
        }
    else:
        return {
            "execution_success": True,
            "execution_error": "",
            "messages": [AIMessage(content=f"Hasil Eksekusi Berhasil:\n{result}")]
        }

# 4. Routing Logic
def router_after_review(state: AgentState):
    return "execute" if state.get("tool_approved", False) else "rewrite"

def router_after_executor(state: AgentState):
    if not state.get("execution_success", True):
        # Jika gagal dan belum melebihi 3 kali percobaan, kirim kembali ke writer secara otomatis
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

# Tambahan conditional edge untuk menangani loop perbaikan otomatis
workflow.add_conditional_edges("executor", router_after_executor, {
    "auto_fix": "writer",            # Kembali ke writer secara otonom untuk memperbaiki error
    "max_retry_exceeded": END,     # Berhenti jika sudah 3 kali gagal terus
    "end": END
})

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)