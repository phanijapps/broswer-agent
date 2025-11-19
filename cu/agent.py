
from cu.utils import get_nomic_embed, get_deepseek_llm, LoggingMiddleware, get_sqlite_checkpointer, RichLoggingMiddleware
from dotenv import load_dotenv
from cu.mem.episodic_memory import EpisodicMemory
from cu.mem.tool_factory import MemoryToolFactory

from langchain.agents.middleware import TodoListMiddleware, SummarizationMiddleware
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver 


from langchain_mcp_adapters.client import MultiServerMCPClient

import uuid
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


import asyncio
import os

load_dotenv()
storage_path = os.getenv("STORAGE_LOCATION", "./mem")

SYSTEM_PROMPT="""
You are a safe, helpful, and versatile AI assistant with integrated browser capabilities. Your goal is to assist users with tasks like web research, navigation, content analysis, or data extraction, while ALWAYS prioritizing ethics, accuracy, privacy, and user well-being. Leverage browser tools to fetch, summarize, and interact with online content when relevant.

Tools:
You have access to browser-focused tools for real-time web interaction:
- Browse Page: Fetch and summarize webpage content based on specific instructions.
- Web Search: Query the internet for results, with options for snippets or operators.
- Additional tools (e.g., code execution, X searches) as needed for comprehensive support.

Core Rules:
- NEVER access, share, or manipulate sensitive/personal data without explicit consent; avoid harmful sites or actions (e.g., phishing simulations, unauthorized logins).
- Before any browser action that could submit data, download files, or follow links, ALWAYS ask for explicit user confirmation and explain risks (e.g., "This will open a login page—confirm?").
- If a query requires web verification, use tools proactively but explain steps; if offline knowledge suffices, note it.
- Prioritize privacy: Do not track, store, or use cookies/sessions beyond the session; suggest incognito mode equivalents.
- Explain reasoning and tool usage in clear, step-by-step language; keep responses concise and focused.
- Limit to the user's explicit query; do not assume or chain unrequested actions.

Respond thoughtfully and engagingly. Use tables for comparisons or lists when effective. End with an offer for follow-up if needed.
"""

async def get_browser_mcp():
    mcp_config = MultiServerMCPClient(
        {
           "chrome-devtools": {
                "transport": "stdio",
                "command": "npx",
                "args": [
                    "-y",
                    "chrome-devtools-mcp@latest",
                    "--browserUrl=http://localhost:9223"
                ]
            }
        }
    )
    return await mcp_config.get_tools()

async def get_playwright_mcp():
    """Connect via docker exec stdio"""
    mcp_config = MultiServerMCPClient(
        {
            "playwright": {
                "transport": "sse",
                "url": "http://localhost:8931/sse",
            
            }
        }
    )
    return await mcp_config.get_tools()

async def build_agent():
    episodic = EpisodicMemory(
        embeddings=get_nomic_embed(),
        persist_dir=storage_path
    )
    memory_tools = MemoryToolFactory(memory=episodic)
    tools = memory_tools.get_tools()

    browser_tools = await get_playwright_mcp()
    tools.extend(browser_tools)

    # Create the context manager
    saver_cm = get_sqlite_checkpointer(storage_path)

    # ENTER IT HERE — THIS RETURNS THE REAL SAVER
    saver = await saver_cm.__aenter__()

    agent = create_agent(
        model=get_deepseek_llm(),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        checkpointer=saver,   # ✔ PASS S A V E R, not CM
        middleware=[
            TodoListMiddleware(),
            SummarizationMiddleware(max_tokens_before_summary=10000,
                                    model=get_deepseek_llm(),
                                    messages_to_keep=6),
            RichLoggingMiddleware(agent_name="Computer Use")
        ]
    )

    return agent, saver_cm

async def repl():

    agent, saver_cm = await build_agent()
    console.print("[bold cyan]💻 Computer Use Agent started. Type 'exit' to quit.[/bold cyan]")

    messages = []
    thread_id = None  # Track thread across turns

    try:
        while True:
            user_input = console.input("[bold green]🧑 You:[/bold green] ").strip()

            # -------------------------------
            # Slash Command Handling
            # -------------------------------
            if user_input.lower() in ("/exit", "/bye", "exit", "quit"):
                console.print("[bold yellow]👋 Exiting. Goodbye![/bold yellow]")
                break

            if user_input.lower() == "/new":
                thread_id = str(uuid.uuid4())
                messages = []
                console.print(f"[bold blue]✨ New thread created:[/bold blue] {thread_id}")
                continue
            # -------------------------------

            if not user_input:
                continue

            # Append user message
            messages.append({"role": "user", "content": user_input})

            # If thread_id is missing, generate it
            if thread_id is None:
                thread_id = str(uuid.uuid4())
                console.print(f"[bold blue]🧵 Starting new thread:[/bold blue] {thread_id}")

            config = {
                "recursion_limit": 100,
                "configurable": {"thread_id": thread_id}
            }

            try:
                result = await agent.ainvoke({"messages": messages}, config=config)
                output = result.get("content", str(result))

             #   console.print(
              #      Panel(
              #          Markdown(output),
              #          title="🤖 AI",
              #          border_style="blue",
              #      )
              #  )

                messages.append({"role": "assistant", "content": output})

            except Exception as e:
                console.print(
                    Panel(
                        Text(str(e), style="bold red"),
                        title="⚠️ Error",
                        border_style="red"
                    )
                )

    finally:
        await saver_cm.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(repl())