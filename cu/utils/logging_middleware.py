"""
Logging Middleware for capturing AI messages and tool calls.
"""
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from typing import Any
import json
from langchain_core.messages import AIMessage
import logging

class LoggingMiddleware(AgentMiddleware):
    def __init__(self, log_level: int = logging.INFO, agent_name: str = "langgraph_agent"):
        self.agent_name = agent_name
        self.logger = logging.getLogger("langgraph_agent")
        self.logger.setLevel(logging.INFO)

    @staticmethod
    def _format_tool_args(args: Any, max_items: int = 1, max_length: int = 200) -> str:
        """Return a concise, truncated string representation of tool args."""
        if args is None:
            return "None"

        # Reduce overly large collections to the first entry when possible.
        truncated: Any = args
        if isinstance(args, dict):
            items = list(args.items())
            truncated = dict(items[:max_items])
            if len(items) > max_items:
                truncated["..."] = f"{len(items) - max_items} more arg(s)"
        elif isinstance(args, (list, tuple)):
            truncated = list(args[:max_items])
            if len(args) > max_items:
                truncated.append("…")

        # Always stringify and trim the payload to avoid noisy logs.
        if isinstance(truncated, str):
            text = truncated
        else:
            try:
                text = json.dumps(truncated, default=str)
            except Exception:
                text = str(truncated)

        if len(text) > max_length:
            return text[:max_length] + "…"
        return text
    
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Log AI responses and tool calls after model execution"""
        if state['messages']:
            last_message = state['messages'][-1]
            
            # Log AI message content
            if isinstance(last_message, AIMessage):
                self.logger.info(f"[{self.agent_name}] AI Response: {last_message.content}")
                
                # Log tool calls if present
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    self.logger.info(f"[{self.agent_name}] Tool calls detected: {len(last_message.tool_calls)}")
                    for tool_call in last_message.tool_calls:
                        formatted_args = self._format_tool_args(tool_call.get('args'))
                        self.logger.info(
                            f"[{self.agent_name}]   Tool: {tool_call['name']}, "
                            f"Args: {formatted_args}, "
                            f"ID: {tool_call['id']}"
                        )
        
        return None
