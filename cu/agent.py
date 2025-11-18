
from cu.utils import get_nomic_embed, get_deepseek_llm, LoggingMiddleware, get_sqlite_checkpointer
from dotenv import load_dotenv
from cu.mem.episodic_memory import EpisodicMemory
from cu.mem.tool_factory import MemoryToolFactory

from langchain.agents.middleware import TodoListMiddleware, SummarizationMiddleware
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver 


from langchain_mcp_adapters.client import MultiServerMCPClient

import uuid



import asyncio
import os

load_dotenv()
storage_path = os.getenv("STORAGE_LOCATION", "./mem")

SYSTEM_PROMPT="""
You are a safe and helpful AI assistant designed for computer use. Your goal is to assist the user with tasks like file management, browsing, or running commands, but ALWAYS prioritize safety and ethics.

Tools:
You are given access to memory and computer tools.

Core Rules:
- NEVER perform harmful, illegal, or unethical actions (e.g., deleting files without explicit confirmation, accessing private data, or running malicious code).
- Before any action that could modify files, install software, or change system settings, ALWAYS ask for explicit user confirmation and explain the risks.
- If unsure about a command or action, refuse and suggest a safer alternative.
- Explain every step you take in plain language.
- Limit actions to the user's explicit instructions; do not assume or improvise.

Respond concisely, and only act when instructed.
"""

async def get_browser_mcp():
    mcp_config = MultiServerMCPClient(
        {
            "playright": {
                "transport": "stdio",
                "command": "npx",
                "args": ["chrome-devtools-mcp@latest"]
            }
        }
    )
    return await mcp_config.get_tools()


async def build_agent():
    """
    Build the agent AND keep the SQLiteSaver context open.
    This must run inside an async function.
    """

    # ---------- Memory ----------
    episodic = EpisodicMemory(
        embeddings=get_nomic_embed(),
        persist_dir=storage_path
    )
    memory_tools = MemoryToolFactory(memory=episodic)
    tools = memory_tools.get_tools()

    # ---------- Browser MCP ----------
    browser_tools = await get_browser_mcp()
    tools.extend(browser_tools)

    # ---------- Checkpointer ----------
    # get_sqlite_checkpointer returns the ASYNC CONTEXT MANAGER
    saver_cm = get_sqlite_checkpointer(storage_path)

    # IMPORTANT:
    # We enter the async context manager and keep it alive for the life of the agent
    saver = await saver_cm.__aenter__()

    # ---------- Build the agent ----------
    agent = create_agent(
        model=get_deepseek_llm(),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        checkpointer=saver,
        middleware=[
            TodoListMiddleware(),
            SummarizationMiddleware(
                max_tokens_before_summary=100000,
                messages_to_keep=6,
                model=get_deepseek_llm()
            ),
            LoggingMiddleware(agent_name="Computer Use")
        ]
    )

    return agent, saver_cm  # return both so we can properly exit later


async def repl():
    """
    Interactive REPL loop with the agent kept alive inside the same async session.
    """

    agent, saver_cm = await build_agent()
    print("💻 Computer Use Agent started. Type 'exit' to quit.")

    messages = []

    try:
        while True:
            user_input = input("🧑 You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("👋 Exiting. Goodbye!")
                break
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            config = {
                "recursion_limit": 30,
                "configurable": {
                    "thread_id": str(uuid.uuid4())
                }
            }

            try:
                result = await agent.ainvoke({"messages": messages}, config=config)
                output = result.get("content", str(result))

                print(f"🤖 Agent: {output}")
                messages.append({"role": "assistant", "content": output})

            except Exception as e:
                print(f"⚠️ Error while running agent: {e}")

    finally:
        # Cleanly close SQLiteSaver context
        await saver_cm.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(repl())