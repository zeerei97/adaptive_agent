# adaptive_agent
Agent can add new tool for the new task
# problem and potentials

**1. PydanticInvalidForJsonSchema:**<br>
Cannot generate a JsonSchema for core_schema.IsInstanceSchema (<class 'langchain_core.runnables.base.Runnable'>)
Error tersebut terjadi karena salah satu fungsi tool di dalam file tools.py memiliki parameter yang menerima objek kompleks LangChain (seperti Runnable).

Ketika LangChain mencoba mendaftarkan fungsi tersebut ke LLM menggunakan .bind_tools(), Pydantic bertugas menerjemahkan parameter fungsi tersebut ke dalam format JSON Schema. Pydantic menolak hal ini karena objek seperti Runnable tidak bisa diubah menjadi format JSON — tool hanya boleh menerima tipe data primitif seperti str, int, float, bool, atau kamus/list sederhana.

Saat kamu menggunakan inspect.getmembers(tools, inspect.isfunction), fungsi tersebut memindai semua fungsi yang ada di dalam modul tools.py, termasuk fungsi atau dekorator yang kamu import dari luar (yaitu tool dari langchain_core.tools).

Karena tool adalah dekorator bawaan LangChain yang memiliki struktur objek internal yang kompleks (termasuk referensi Runnable), Pydantic gagal mengubahnya menjadi JSON Schema saat dicoba untuk di-bind ke LLM.
Cara Memperbaikinya
Karena jumlah tools di dalam file tools.py biasanya sudah pasti dan terstruktur, cara paling aman, bersih, dan direkomendasikan adalah mengimpor tools tersebut secara langsung alih-alih menggunakan pemindaian otomatis inspect.

Ubah cara kamu memuat tools di file utama atau notebook-mu menjadi seperti ini:
# Impor langsung fungsi tool yang sudah didekorasi di tools.py
from tools import add_numbers

# Masukkan ke dalam list tools secara eksplisit
tools = [add_numbers]

# **status: TERMINATED**

# Bind ke LLM
llm_with_tools = llm.bind_tools(tools)
print("Tools berhasil dimuat:", [t.name for t in tools])

**2. too dangerous to give the key car to the autonomus car**
