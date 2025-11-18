from enum import Enum
import os
from typing import Any, Callable, Dict, Union

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()  # Load environment variables from .env file

class ModelProvider(Enum):
    """Enumeration of supported LLM model providers with their configurations."""
    OLLAMA = {
        "provider": "ollama", 
        "api_base": "http://localhost:11434", 
        "default_model": "llama3.2:3b"
    }
    OLLAMA_CLOUD = {
        "provider": "ollama_cloud", 
        "api_base": "https://ollama.com", 
        "default_model": "kimi-k2:1t-cloud"
    }
    OPENAI = {
        "provider": "openai",
        "api_base": "https://api.openai.com/v1",
        "default_model": "gpt-4"
    }
    DEEPSEEK = {
        "provider": "deepseek",
        "api_base": "https://api.deepseek.com/v1",
        "default_model": "deepseek-reasoner"
    }
    OPENROUTER = {
        "provider": "openrouter",
        "api_base": "https://openrouter.ai/api/v1",
        "default_model": "anthropic/claude-sonnet-4"
    }
    OTHER = {
        "provider": "other",
        "api_base": None,
        "default_model": None
    }

class LLMFactory:
    """Factory class to create LLM instances based on the specified model provider."""
    _PROVIDER_BUILDERS: Dict[
        ModelProvider, Callable[[Dict[str, Any], str | None], Union[ChatOllama, Any]]
    ] = {}

    @staticmethod
    def _require_env_var(env_var: str) -> str:
        value = os.getenv(env_var)
        if not value:
            raise ValueError(f"{env_var} environment variable not set.")
        return value

    @classmethod
    def _build_ollama(cls, config: Dict[str, Any], model_name: str | None) -> ChatOllama:
        model_to_use = model_name or config["default_model"]
        return ChatOllama(model=model_to_use, base_url=config["api_base"])

    @classmethod
    def _build_ollama_cloud(cls, config: Dict[str, Any], model_name: str | None) -> ChatOllama:
        api_key = cls._require_env_var("OLLAMA_API_KEY")
        model_to_use = model_name or config["default_model"]
        return ChatOllama(model_name=model_to_use, api_key=api_key, base_url=config["api_base"])

    @classmethod
    def _build_openai_like(
        cls,
        *,
        config: Dict[str, Any],
        model_name: str | None,
        env_var: str,
        extra_kwargs: Dict[str, Any] | None = None,
    ) -> ChatOpenAI:
        model_to_use = model_name or config["default_model"]
        api_key = cls._require_env_var(env_var)

        kwargs: Dict[str, Any] = {
            "model_name": model_to_use,
            "openai_api_key": api_key,
            "temperature": 0.7,
            "max_completion_tokens": 5000,
        }
        if config.get("api_base"):
            kwargs["openai_api_base"] = config["api_base"]
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return ChatOpenAI(**kwargs)

    @classmethod
    def _build_openai(cls, config: Dict[str, Any], model_name: str | None) -> ChatOpenAI:
        return cls._build_openai_like(config=config, model_name=model_name, env_var="OPENAI_API_KEY")

    @classmethod
    def _build_deepseek(cls, config: Dict[str, Any], model_name: str | None) -> ChatOpenAI:
        return cls._build_openai_like(config=config, model_name=model_name, env_var="DEEPSEEK_API_KEY")

    @classmethod
    def _build_openrouter(cls, config: Dict[str, Any], model_name: str | None) -> ChatOpenAI:
        return cls._build_openai_like(
            config=config,
            model_name=model_name,
            env_var="OPENROUTER_API_KEY",
            extra_kwargs={"timeout": 300},
        )

    @classmethod
    def _get_builder(cls, provider: ModelProvider) -> Callable[[Dict[str, Any], str | None], Any] | None:
        if not cls._PROVIDER_BUILDERS:
            cls._PROVIDER_BUILDERS = {
                ModelProvider.OLLAMA: cls._build_ollama,
                ModelProvider.OLLAMA_CLOUD: cls._build_ollama_cloud,
                ModelProvider.OPENAI: cls._build_openai,
                ModelProvider.DEEPSEEK: cls._build_deepseek,
                ModelProvider.OPENROUTER: cls._build_openrouter,
            }
        return cls._PROVIDER_BUILDERS.get(provider)

    @classmethod
    def get_llm_model(
        cls,
        model_name: str = None,
        model_provider: ModelProvider = ModelProvider.OLLAMA,
    ) -> Union[ChatOllama, Any]:
        """Returns an instance of the specified LLM model based on the provider."""
        provider_config = model_provider.value
        builder = cls._get_builder(model_provider)
        if builder is None:
            raise ValueError(f"Unsupported model provider: {provider_config['provider']}")
        return builder(provider_config, model_name)
