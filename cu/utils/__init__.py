from dotenv import load_dotenv
from cu.utils.embedding_factory import get_embeddings_by_name
from cu.utils.llm_factory import LLMFactory, ModelProvider
from cu.utils.logging_middleware import LoggingMiddleware, RichLoggingMiddleware
import sqlite3
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver 
from typing import Optional

load_dotenv()

_nomic_embed = None
_deepseek_llm = None

def get_nomic_embed():
    """Get Nomic embeddings (lazy loaded)."""
    global _nomic_embed
    if _nomic_embed is None:
        _nomic_embed = get_embeddings_by_name("ollama")
    return _nomic_embed

def get_deepseek_llm():
    """Get DeepSeek LLM (lazy loaded)."""
    global _deepseek_llm
    if _deepseek_llm is None:
        _deepseek_llm = LLMFactory.get_llm_model(model_provider=ModelProvider.DEEPSEEK)
    return _deepseek_llm



def get_sqlite_checkpointer(
    db_path: str = "./mem"
) -> AsyncSqliteSaver:

    conn_string = f"file:{db_path}/checkpointer.db"

    # NOTE: sqlite3 connect is okay but not required for AsyncSqliteSaver
    # You can remove the manual connect entirely — AsyncSqliteSaver handles it.

    saver = AsyncSqliteSaver.from_conn_string(conn_string=conn_string)
    return saver