# Browser use agent with Chrome Dev Tools MCP and Langchain

Small LangGraph/LangChain agent that drives the browser through the Chrome DevTools MCP client. Uses DeepSeek for the LLM, Ollama embeddings for memory, and stores state in SQLite/Chroma under `./mem`.

## What you need
- Python 3.13+
- `uv` CLI (`pip install uv`)
- Chrome installed (for the MCP tool)
- Ollama running locally with the `nomic-embed-text` model, or change the embedding provider in `cu/utils/__init__.py`

## Setup and run
1) Copy `.env_example` to `.env` and set at least `DEEPSEEK_API_KEY`. `LANGSMITH_API_KEY` is optional; `OPENROUTER_API_KEY` is only needed if you switch providers.
2) From the repo root run `./run.sh` (Linux/Mac) or `./run.ps1` (Windows). The script will create `.venv` with `uv venv --python=3.13` if missing, install `requirements.txt`, and start `python -m cu.agent`.
3) Type your requests in the REPL; type `exit` to quit.

## Notes
- Set `STORAGE_LOCATION` in `.env` if you want memory/checkpoints in a different folder.
- If `uv` is missing, install it first (`pip install uv`) before running the scripts.


## Experimental
```bash

docker build -t chrome-mcp-debian:latest .


docker run -d \
  --name chrome-mcp-container \
  -p 5901:5901 \
  -p 6901:6901 \
  -p 9223:9223 \
  chrome-mcp-debian:latest
```