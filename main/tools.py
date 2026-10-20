from langchain_core.tools import tool, BaseTool

@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two integers together."""
    return a + b