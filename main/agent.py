import os
import math
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from RestrictedPython import compile_restricted, safe_globals

# Konfigurasi API Key Gemini (pastikan GEMINI_API_KEY sudah diset)
# os.environ["GEMINI_API_KEY"] = "your-api-key-here"

# 1. Definisi State Graph
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    generated_code: str
    tool_approved: bool
    feedback: str
load_dotenv()

# Inisialisasi LLM Gemini
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

# ---------------------------------------------------------
# Utility Helpers & RestrictedPython Sandbox
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

def run_with_restricted_python(code_text: str) -> str:
    """
    Sandbox menggunakan RestrictedPython:
    - Mengompilasi kode dan memblokir akses berbahaya (akses file, import bebas, atribut privat).
    - Menyediakan lingkungan globals yang bersih (safe_globals) + modul math.
    - Menangkap fungsi print() secara aman tanpa mencetak langsung ke sistem utama.
    """
    # 1. Kompilasi kode dengan pembatasan keamanan
    byte_code = compile_restricted(code_text, '<string>', 'exec')
    
    if byte_code.errors:
        return f"Ditolak oleh kebijakan keamanan RestrictedPython: {byte_code.errors}"

    # 2. Siapkan lingkungan aman
    my_globals = safe_globals.copy()
    my_globals['math'] = math

    # Penampung hasil cetak (print collector) yang aman
    printed_output = []
    def safe_print(*args, **kwargs):
        output = " ".join(map(str, args))
        printed_output.append(output)

    # RestrictedPython otomatis mengarahkan fungsi print() ke _print
    my_globals['_print'] = safe_print
    
    my_locals = {}

    try:
        # 3. Eksekusi bytecode yang sudah dibersihkan
        exec(byte_code.code, my_globals, my_locals)
        result = "\n".join(printed_output)
        return result if result else "Eksekusi berhasil (tanpa output print)."
    except Exception as e:
        return f"Error saat eksekusi runtime: {e}"

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
1. Gunakan modul `math` jika diperlukan (misal: math.factorial, math.sqrt, dll).
2. Cetak hasil akhir perhitungan menggunakan fungsi `print(...)`.
3. Keluarkan HANYA blok kode Python valid di dalam markdown ```python ... ``` tanpa teks penjelasan lain.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    code_text = extract_text(response.content)
    
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
# Node 3: Code Executor (RestrictedPython Sandbox)
# ---------------------------------------------------------
def code_executor_node(state: AgentState):
    """Mengeksekusi kode di dalam sandbox RestrictedPython yang aman."""
    code = state["generated_code"]
    result = run_with_restricted_python(code)
    
    return {"messages": [AIMessage(content=f"Hasil Eksekusi RestrictedPython:\n{result}")]}

# ---------------------------------------------------------
# Routing Logic
# ---------------------------------------------------------
def router_after_review(state: AgentState):
    if state.get("tool_approved", False):
        return "execute"
    else:
        return "rewrite"

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

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "restricted_session_1"}}

# 1. Kirim prompt awal
query = HumanMessage(content="Tolong hitung nilai dari 10 faktorial dikurangi akar kuadrat dari 10000.")
print(f"User: {query.content}\n" + "-"*50)

for event in app.stream({"messages": [query]}, config, stream_mode="values"):
    if "messages" in event:
        latest = event["messages"][-1]
        if isinstance(latest, AIMessage) and latest.content:
            print(f"Agent: {latest.content}")

# 2. Periksa status interrupt
state = app.get_state(config)
if state.next:
    print("\n[Graph Berhenti: Menunggu Peninjauan Manusia]")
    if state.tasks and state.tasks[0].interrupts:
        interrupt_value = state.tasks[0].interrupts[0].value
        print("\n--- KODE PYTHON YANG DIBUAT AGEN ---")
        print(interrupt_value["generated_code"])
        print("-" * 50)
        
        # 3. Simulasi Manusia Menyetujui Kode
        human_response = {
            "approved": True, 
            "feedback": "Kode aman."
        }
        
        print("\nManusia menyetujui kode. Menjalankan di RestrictedPython...")
        
        for event in app.stream(Command(resume=human_response), config, stream_mode="values"):
            if "messages" in event:
                latest = event["messages"][-1]
                if isinstance(latest, AIMessage) and latest.content:
                    print(f"\n{latest.content}")
