import os
import sys
import io
import math
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv()

# Konfigurasi API Key Gemini (pastikan GEMINI_API_KEY sudah diset di environment variables)
# os.environ["GEMINI_API_KEY"] = "your-api-key-here"
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

# 1. Definisi State Graph
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    generated_code: str
    tool_approved: bool
    feedback: str

# ---------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------
def extract_text(content: Any) -> str:
    """Mengekstrak string murni dari berbagai format respons Gemini"""
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

def safe_execute_code(code_text: str) -> str:
    """
    Sandbox exec() yang aman dan terisolasi:
    - Membatasi built-in functions yang berbahaya (tidak ada open, exec, import bebas).
    - Hanya menyediakan fungsi dasar yang aman dan modul `math`.
    - Menangkap hasil print() untuk dikembalikan sebagai string.
    """
    safe_builtins = {
        "abs": abs, "max": max, "min": min, "pow": pow, "round": round,
        "sum": sum, "range": range, "int": int, "float": float,
        "str": str, "list": list, "dict": dict, "len": len, "print": print
    }

    restricted_globals = {
        "__builtins__": safe_builtins,
        "math": math
    }

    local_namespace = {}
    
    # Redirect stdout untuk menangkap hasil print()
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        exec(code_text, restricted_globals, local_namespace)
        output_result = new_stdout.getvalue()
    except Exception as e:
        output_result = f"Error saat eksekusi kode: {e}"
    finally:
        sys.stdout = old_stdout

    return output_result.strip()

# ---------------------------------------------------------
# Node 1: Code Writer (Agen Pembuat Kode)
# ---------------------------------------------------------
def code_writer_node(state: AgentState):
    """Agen yang menulis skrip Python untuk menyelesaikan masalah matematika."""
    messages = state["messages"]
    feedback = state.get("feedback", "")
    last_message = extract_text(messages[-1].content)
    
    prompt = f"""
Kamu adalah seorang Python Code Interpreter profesional.
Permintaan pengguna: "{last_message}"
{f"Feedback penolakan sebelumnya dari manusia: {feedback}" if feedback else ""}

Tuliskan kode Python lengkap yang menyelesaikan perhitungan matematika di atas.
Ketentuan:
1. Gunakan modul `math` jika diperlukan.
2. Cetak hasil akhir perhitungan menggunakan fungsi `print(...)`.
3. Keluarkan HANYA blok kode Python valid di dalam markdown ```python ... ``` tanpa teks penjelasan lain di luar blok kode.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    code_text = extract_text(response.content)
    
    # Ekstraksi blok kode dari markdown
    if "```python" in code_text:
        code_text = code_text.split("```python")[1].split("```")[0].strip()
    elif "```" in code_text:
        code_text = code_text.split("```")[1].split("```")[0].strip()
        
    return {"generated_code": code_text}

# ---------------------------------------------------------
# Node 2: Human Review & Approval (Interrupt)
# ---------------------------------------------------------
def human_review_node(state: AgentState):
    """Menjeda eksekusi graph untuk meminta persetujuan manusia."""
    code = state["generated_code"]
    
    # LangGraph interrupt untuk human-in-the-loop
    human_decision = interrupt({
        "prompt": "Agen telah menulis kode Python baru. Mohon tinjau kodenya:",
        "generated_code": code,
        "instructions": "Ketik 'approved': True untuk menyetujui, atau 'approved': False dengan 'feedback' untuk merevisi."
    })
    
    approved = human_decision.get("approved", False)
    feedback = human_decision.get("feedback", "")
    
    return {
        "tool_approved": approved,
        "feedback": feedback
    }

# ---------------------------------------------------------
# Node 3: Code Executor (Sandbox Execution)
# ---------------------------------------------------------
def code_executor_node(state: AgentState):
    """Mengeksekusi kode yang sudah disetujui di dalam sandbox yang aman."""
    code = state["generated_code"]
    result = safe_execute_code(code)
    
    return {"messages": [AIMessage(content=f"Hasil Eksekusi Program:\n{result}")]}

# ---------------------------------------------------------
# Routing Logic
# ---------------------------------------------------------
def router_after_review(state: AgentState):
    if state.get("tool_approved", False):
        return "execute"
    else:
        return "rewrite" # Kembali ke writer jika ditolak untuk diperbaiki berdasarkan feedback

# ---------------------------------------------------------
# Menyusun Workflow LangGraph
# ---------------------------------------------------------
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
workflow.add_edge("executor", END)

# Wajib menggunakan MemorySaver untuk mendukung fitur interrupt
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Konfigurasi thread untuk sesi interaksi
config = {"configurable": {"thread_id": "sandbox_session_1"}}

# 1. Kirim prompt awal ke agent
query = HumanMessage(content="Tolong hitung nilai dari 8 faktorial ditambah akar kuadrat dari 144.")
print(f"User: {query.content}\n" + "-"*50)

# Jalankan graph sampai mencapai titik interrupt (human_review)
for event in app.stream({"messages": [query]}, config, stream_mode="values"):
    if "messages" in event:
        latest = event["messages"][-1]
        if isinstance(latest, AIMessage) and latest.content:
            print(f"Agent: {latest.content}")

# 2. Cek status graph apakah sedang berhenti di interrupt
state = app.get_state(config)
if state.next:
    print("\n[Graph Berhenti: Menunggu Peninjauan & Persetujuan Manusia]")
    
    # Ambil data kode yang di-generate agent dari payload interrupt
    if state.tasks and state.tasks[0].interrupts:
        interrupt_value = state.tasks[0].interrupts[0].value
        print("\n--- KODE PYTHON YANG DIBUAT AGEN ---")
        print(interrupt_value["generated_code"])
        print("-" * 50)
        
        # 3. Simulasi Keputusan Manusia (Contoh: Menyetujui kode)
        # Jika ingin menolak, ganti 'approved': False dan berikan 'feedback'
        human_response = {
            "approved": True, 
            "feedback": "Kode sudah benar dan aman."
        }
        
        print("\nManusia menyetujui kode. Melanjutkan eksekusi di sandbox...")
        
        # Lanjutkan graph menggunakan Command(resume=...)
        for event in app.stream(Command(resume=human_response), config, stream_mode="values"):
            if "messages" in event:
                latest = event["messages"][-1]
                if isinstance(latest, AIMessage) and latest.content:
                    print(f"\n{latest.content}")
