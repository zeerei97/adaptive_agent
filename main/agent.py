import os
import sys
import importlib.util
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
# Konfigurasi API Key Gemini (pastikan GEMINI_API_KEY sudah diset di environment variables)
# os.environ["GEMINI_API_KEY"] = "your-api-key-here"

TOOLS_FILE = "dynamic_math_tools.txt"

# 1. State Definition
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    missing_math_op: str       # Operasi matematika yang belum ada tool-nya
    generated_tool_code: str   # Kode Python tool yang dibuat LLM
    tool_approved: bool        # Status persetujuan manusia
    feedback: str              # Catatan/alasan jika ditolak

# Inisialisasi Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

# ---------------------------------------------------------
# Utility: Manajemen File .txt untuk Tool Dinamis
# ---------------------------------------------------------
def save_tool_to_file(code: str):
    """Menyimpan kode tool yang disetujui ke file .txt"""
    with open(TOOLS_FILE, "a", encoding="utf-8") as f:
        f.write("\n\n# --- DYNAMIC TOOL ---\n")
        f.write(code)

def load_dynamic_tools() -> Dict[str, callable]:
    """Memuat fungsi-fungsi tool dari file .txt saat runtime"""
    loaded_tools = {}
    if not os.path.exists(TOOLS_FILE):
        return loaded_tools
    
    spec = importlib.util.spec_from_file_location("dynamic_math_tools", TOOLS_FILE)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules["dynamic_math_tools"] = module
        try:
            spec.loader.exec_module(module)
            # Ambil semua fungsi yang didefinisikan di module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if callable(attr) and not attr_name.startswith("_"):
                    loaded_tools[attr_name] = attr
        except Exception as e:
            print(f"Error memuat dynamic tools: {e}")
            
    return loaded_tools

# ---------------------------------------------------------
# Node 1: Agent Router & Decision Maker
# ---------------------------------------------------------
# 1. Modifikasi Agent Node agar mendeteksi tool yang tersedia dan membind-nya ke LLM
def math_agent_node(state: AgentState):
    """Agent utama yang dilengkapi tool dinamis yang sudah disetujui."""
    messages = state["messages"]
    
    # Muat tool dinamis dari file .txt
    dynamic_tools_dict = load_dynamic_tools()
    tools_list = list(dynamic_tools_dict.values())
    
    # Bind tool ke Gemini agar LLM bisa melakukan Native Tool Calling
    bound_llm = llm.bind_tools(tools_list) if tools_list else llm
    
    system_prompt = SystemMessage(content="""
Kamu adalah agent matematika khusus. 
ATURAN KETAT:
1. Kamu Wajib menggunakan tool yang tersedia untuk menghitung operasi matematika.
2. Jika tool yang dibutuhkan BELUM ADA di daftar tool yang terikat, jangan dihitung manual. Berikan respons khusus dengan format persis:
   [MISSING_TOOL: nama_fungsi (deskripsi singkat)]
   Contoh: [MISSING_TOOL: hitung_faktorial (Menghitung nilai faktorial dari angka n)]
""")
    
    response = bound_llm.invoke([system_prompt] + messages)
    return {"messages": [response]}

# ---------------------------------------------------------
# Node 2: Tool Creator (Membuat tool baru jika belum ada)
# ---------------------------------------------------------
def tool_creator_node(state: AgentState):
    """Agent pembuat tool yang menulis kode Python bersih untuk operasi matematika baru."""
    messages = state["messages"]
    last_message = messages[-1].content
    feedback = state.get("feedback", "")
    
    creation_prompt = f"""
Seorang user meminta perhitungan matematika, namun tool yang dibutuhkan belum tersedia.
Permintaan/Konteks terakhir: {last_message}
{f"Feedback dari penolakan sebelumnya: {feedback}" if feedback else ""}

Tuliskan kode fungsi Python (menggunakan decorator `@tool` dari `langchain_core.tools` atau fungsi Python standar dengan type hinting lengkap dan docstring yang jelas) untuk menyelesaikan masalah tersebut.
Keluarkan HANYA blok kode Python valid tanpa teks penjelasan di luar blok kode.
"""
    response = llm.invoke([HumanMessage(content=creation_prompt)])
    
    # Ekstraksi kode Python dari respons LLM
    code_text = response.content
    if "```python" in code_text:
        code_text = code_text.split("```python")[1].split("```")[0].strip()
    elif "```" in code_text:
        code_text = code_text.split("```")[1].split("```")[0].strip()
        
    return {"generated_tool_code": code_text}

# ---------------------------------------------------------
# Node 3: Human Review & Approval (Breakpoint / Interrupt)
# ---------------------------------------------------------
def human_review_node(state: AgentState):
    """Menyerahkan tool ke manusia untuk disetujui atau ditolak menggunakan interrupt LangGraph."""
    code = state["generated_tool_code"]
    
    # Melakukan interrupt untuk meminta input persetujuan dari manusia
    human_decision = interrupt({
        "prompt": "Tool matematika baru telah dibuat oleh agent. Mohon tinjau kode di bawah ini:",
        "generated_code": code,
        "instructions": "Ketik 'approve' untuk menyetujui dan menyimpan, atau berikan alasan penolakan untuk merevisi."
    })
    
    # human_decision berisi dictionary atau nilai yang dikirimkan saat meresume graph
    approved = human_decision.get("approved", False)
    feedback = human_decision.get("feedback", "")
    
    if approved:
        save_tool_to_file(code)
        
    return {
        "tool_approved": approved,
        "feedback": feedback
    }

# ---------------------------------------------------------
# Node 4: Tool Execution
# ---------------------------------------------------------
# 3. Tool Executor Node yang benar-benar mengeksekusi fungsi Python secara programmatik
def tool_executor_node(state: AgentState):
    """Mengeksekusi fungsi Python dari file .txt berdasarkan tool_calls dari Gemini."""
    dynamic_tools_dict = load_dynamic_tools()
    
    # Menggunakan ToolNode bawaan LangGraph untuk mengeksekusi tool yang di-bind
    from langgraph.prebuilt import ToolNode
    tool_node = ToolNode(list(dynamic_tools_dict.values()))
    
    return tool_node.invoke(state)

# ---------------------------------------------------------
# Routing Logic
# ---------------------------------------------------------
# 2. Modifikasi Routing untuk mendeteksi apakah agent ingin memanggil tool atau butuh tool baru
def router_after_agent(state: AgentState):
    last_msg = state["messages"][-1]
    
    # Jika LLM menghasilkan tool_calls, arahkan ke executor tool
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "execute_tool"
        
    # Jika LLM mengeluarkan teks [MISSING_TOOL:...], buat tool baru
    if isinstance(last_msg.content, str) and "[MISSING_TOOL:" in last_msg.content:
        return "create_tool"
        
    return "end"

def router_after_review(state: AgentState):
    if state.get("tool_approved", False):
        return "execute_tool"
    else:
        return "create_tool" # Rekomendasi ulang jika ditolak

# ---------------------------------------------------------
# Membangun LangGraph Workflow
# ---------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("agent", math_agent_node)
workflow.add_node("tool_creator", tool_creator_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("tool_executor", tool_executor_node) # Menggunakan eksekusi programmatik

workflow.add_edge(START, "agent")

# Routing setelah agent
workflow.add_conditional_edges("agent", router_after_agent, {
    "execute_tool": "tool_executor",
    "create_tool": "tool_creator",
    "end": END
})

workflow.add_edge("tool_creator", "human_review")
workflow.add_conditional_edges("human_review", router_after_review, {
    "create_tool": "tool_creator",
    "execute_tool": "tool_executor"
})
workflow.add_edge("tool_executor", "agent") # Kembalikan ke agent agar LLM bisa membacakan hasil tool ke user

# Compile graph dengan memory checkpointer (diperlukan untuk fitur interrupt)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Konfigurasi thread untuk interaksi human-in-the-loop
config = {"configurable": {"thread_id": "math_session_1"}}

# 1. User memberikan perintah operasi matematika yang tool-nya belum ada
query = HumanMessage(content="Tolong hitung faktorial dari angka 6 menggunakan tool.")
print(f"User: {query.content}\n" + "-"*40)

# Jalankan agent hingga mencapai titik interrupt (human_review)
for event in app.stream({"messages": [query]}, config, stream_mode="values"):
    if "messages" in event:
        latest = event["messages"][-1]
        if isinstance(latest, AIMessage) and latest.content:
            print(f"Agent: {latest.content}")

# Cek apakah graph berhenti di interrupt
state = app.get_state(config)
if state.next:
    print("\n[Graph Berhenti: Menunggu Persetujuan Manusia]")
    current_node = state.next[0]
    print(f"Berhenti di node: {current_node}")
    
    if current_node == "human_review":
        # Ambil task interrupt untuk melihat kode yang digenerate agent
        task_interrupt = state.tasks[0].interrupts[0].value
        print("\n--- KODE TOOL YANG DIBUAT AGENT ---")
        print(task_interrupt["generated_code"])
        print("-" * 40)
        
        # 2. Simulasi Keputusan Manusia (Contoh: Menyetujui tool)
        # Jika ingin menolak, ubah approved=False dan berikan feedback.
        human_response = {"approved": True, "feedback": "Kode sudah bagus dan aman."}
        
        print("\nManusia menyetujui tool. Melanjutkan eksekusi...")
        
        # Lanjutkan graph dengan mengirim hasil keputusan manusia menggunakan Command/resume
        from langgraph.types import Command
        for event in app.stream(Command(resume=human_response), config, stream_mode="values"):
            if "messages" in event:
                latest = event["messages"][-1]
                if isinstance(latest, AIMessage) and latest.content:
                    print(f"Agent Final: {latest.content}")