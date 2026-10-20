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

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
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

# --- 3. DEFINITION OF ALL NODE AGENTS ---

def planner_node(state: AgentState):
    """
    Brain Nodes: Analyze requests, select tools, or decide to create new tools.
    """
    print("--- [Planner] Thinking... ---")
    messages = state["messages"]
    tools = state["available_tools"]
    
    # Bind tools yang tersedia saat ini ke LLM
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

# --- 4. FILTER GUARDRAIL AST (PENGAMAN) ---
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
                    return False, f"Dilarang import modul: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in FORBIDDEN_MODULES:
                return False, f"Dilarang import dari: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
                return False, f"Dilarang panggil fungsi: {node.func.id}()"
    return True, "Aman"

# --- 5. TOOL MAKER NODE DENGAN PENYIMPANAN PERMANEN ---
def persistent_tool_maker_node(state: AgentState):
    print("--- [Tool Maker] Membuat Tool Baru & Menyimpannya ke File... ---")
    messages = state["messages"]
    last_message = messages[-1]
    
    # 1. Minta LLM membuat kode fungsi Python lengkap dengan dekorator @tool
    maker_prompt = SystemMessage(content=(
        "You are an expert Python developer."
        "Write a clean Python function using the @tool decorator from langchain_core.tools."
        "Include clear type hints and docstrings."
        "ONLY output valid Python code without markdown code blocks."
    ))
    
    response = llm.invoke([maker_prompt, HumanMessage(content=f"Create a tool for: {last_message.content}")])
    generated_code = response.content.strip()
    
    # Bersihkan markdown jika LLM tak sengaja menyertainya (` ```python ... ``` `)
    if generated_code.startswith("```"):
        generated_code = generated_code.split("```")[1]
        if generated_code.startswith("python"):
            generated_code = generated_code[6:].strip()
            
    # 2. Filter / Validasi Keamanan AST
    is_safe, msg = validate_code_safety(generated_code)
    if not is_safe:
        print(f"[SECURITY REJECTED] Kode ditolak karena alasan keamanan: {msg}")
        return {"messages": [AIMessage(content=f"Pembuatan tool ditolak oleh sistem keamanan: {msg}")]}
    
    # 3. Simpan secara permanen ke file 'tools.py'
    try:
        with open("tools.py", "a", encoding="utf-8") as f:
            f.write(f"\n\n{generated_code}\n")
        print("[Tool Maker] Berhasil menambahkan tool baru ke file 'tools.py'!")
        
        # 4. Hot-Reload modul 'tools' agar agen bisa langsung memakainya tanpa restart program
        importlib.reload(tools)
        
        # Ambil ulang semua fungsi tool yang ada di file tools.py yang telah diperbarui
        updated_tools_list = [
            obj for name, obj in inspect.getmembers(tools, inspect.isfunction) 
            if hasattr(obj, "tool") or hasattr(obj, "args_schema") # Sesuaikan dengan dekorator LangChain
        ]
        
        return {"available_tools": updated_tools_list}
        
    except Exception as e:
        print(f"Gagal menyimpan ke file: {e}")
        return {"messages": [AIMessage(content=f"Gagal menyimpan tool ke file: {e}")]}

def refresh_available_tools() -> list:
    """
    Memindai file tools.py secara otomatis untuk mengambil 
    semua fungsi yang berstatus sebagai tool.
    """
    # 1. Reload modul agar memuat kode terbaru dari file fisik
    importlib.reload(tools)
    
    refreshed_tools = []
    
    # 2. Ambil semua anggota (fungsi/variabel) di dalam modul tools.py
    for name, obj in inspect.getmembers(tools, inspect.isfunction):
        # Cek apakah objek tersebut adalah tool LangChain 
        # (biasanya memiliki atribut 'tool_call' atau merupakan instance BaseTool)
        if hasattr(obj, "name") and hasattr(obj, "invoke"):
            refreshed_tools.append(obj)
            
    return refreshed_tools

def execution_node(state: AgentState):
    """
    Node Eksekutor: Menjalankan call tool yang diminta oleh Planner.
    """
    print("--- [Eksekutor] Menjalankan Tool... ---")
    messages = state["messages"]
    last_message = messages[-1]
    tools = state["available_tools"]
    
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    
    # Cari fungsi tool berdasarkan nama
    chosen_tool = next((t for t in tools if t.name == tool_name), None)
    
    if chosen_tool:
        result = chosen_tool.invoke(tool_args)
        print(f"[Eksekutor] Hasil {tool_name}({tool_args}): {result}")
        # Kembalikan sebagai AIMessage dengan tool_call_id
        return {"messages": [AIMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_name)]}
    else:
        return {"messages": [AIMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call["id"])]}

# --- 6. DEFENISI LOGIKA PENGARAH (ROUTER) ---

def router(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # 1. Jika Planner meminta panggil tool
    if last_message.tool_calls:
        return "call_tool"
    
    # 2. Jika Planner secara eksplisit meminta buat tool baru dalam teksnya
    if "new tool" in last_message.content.lower() or "create a tool" in last_message.content.lower():
        return "make_tool"
    
    # 3. Jika tidak ada tindakan lanjutan, akhiri percakapan
    return END

# --- 6. MEMBANGUN GRAPH (LANGGRAPH) ---

workflow = StateGraph(AgentState)

# Tambahkan Node
workflow.add_node("planner", planner_node)
workflow.add_node("tool_maker", persistent_tool_maker_node)
workflow.add_node("executor", execution_node)

# Tentukan Alur Kontrol (Edges & Routers)
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

# Setelah membuat tool, kembali ke Planner untuk mengevaluasi ulang
workflow.add_edge("tool_maker", "planner")
# Setelah menjalankan tool, kembali ke Planner untuk merangkum jawaban
workflow.add_edge("executor", "planner")

# Compile Graph
app = workflow.compile()

# --- 7. PENGUJIAN AGEN ---

if __name__ == "__main__":
    print("\n=== MEMULAI AGEN MATEMATIKA ADAPTIF & INTERAKTIF ===\n")
    print("Ketik pertanyaan matematika (contoh: 'Berapa 10 ditambah 5' atau '15 dikurangi 7').")
    print("Ketik 'exit' atau 'keluar' untuk berhenti.\n")
    
    # 1. Inisialisasi state awal dengan memuat semua tool dari file tools.py secara otomatis
    current_state = {
        "messages": [],
        "available_tools": refresh_available_tools()
    }
    
    # 2. Perulangan interaktif untuk menerima input pengguna
    while True:
        try:
            user_input = input("\nUser: ").strip()
            
            if user_input.lower() in ["exit", "keluar"]:
                print("Keluar dari program. Sampai jumpa!")
                break
                
            if not user_input:
                continue
                
            # Tambahkan pesan baru dari pengguna ke dalam riwayat messages di state
            current_state["messages"].append(HumanMessage(content=user_input))
            
            # Jalankan graph secara utuh. 
            # Agen (Planner) akan otomatis memutuskan apakah pakai tool lama, buat tool baru, atau langsung menjawab.
            print("\n--- [Sistem] Agen sedang memproses... ---")
            current_state = app.invoke(current_state)
            
            # Ambil jawaban terakhir dari agen
            final_response = current_state["messages"][-1].content
            print(f"\nAI Agent: {final_response}")
            
            # Perbarui daftar tool di state (jika node tool_maker sempat menambahkan tool baru ke tools.py)
            current_state["available_tools"] = refresh_available_tools()
            
        except KeyboardInterrupt:
            print("\nProgram dihentikan oleh pengguna.")
            break
        except Exception as e:
            print(f"\nTerjadi error: {e}")