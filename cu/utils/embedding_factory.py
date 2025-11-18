"""
EmbeddingFactory: Production-Ready Embeddings Provider Factory

Parallel design to LLMFactory for flexible embeddings provider selection.
Supports: OpenAI, HuggingFace, Ollama, Bedrock, Azure, Cohere, Custom

Usage:
    from embedding_factory import EmbeddingFactory, EmbeddingProvider

    # Option 1: Local (default)
    embeddings = EmbeddingFactory.get_embeddings()

    # Option 2: Specific provider
    embeddings = EmbeddingFactory.get_embeddings(
        provider=EmbeddingProvider.OPENAI,
        model_name="text-embedding-3-small"
    )

    # Option 3: Environment-based
    embeddings = EmbeddingFactory.get_embeddings(
        provider=EmbeddingProvider.OPENAI
    )  # Uses OPENAI_API_KEY from .env
"""

from enum import Enum
import os
from typing import Any, Callable, Dict, Union, Optional
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
import logging


load_dotenv()  # Load environment variables from .env file


class EmbeddingProvider(Enum):
    """Enumeration of supported embedding model providers with their configurations."""

    HUGGINGFACE = {
        "provider": "huggingface",
        "api_base": None,
        "default_model": "all-MiniLM-L6-v2"
    }

    OPENAI = {
        "provider": "openai",
        "api_base": "https://api.openai.com/v1",
        "default_model": "text-embedding-3-small"
    }

    OLLAMA = {
        "provider": "ollama",
        "api_base": "http://localhost:11434",
        "default_model": "nomic-embed-text"
    }

    OLLAMA_CLOUD = {
        "provider": "ollama_cloud",
        "api_base": "https://ollama.com",
        "default_model": "nomic-embed-text"
    }

    BEDROCK = {
        "provider": "bedrock",
        "api_base": None,
        "default_model": "amazon.titan-embed-text-v2:0"
    }

    AZURE = {
        "provider": "azure",
        "api_base": None,
        "default_model": "text-embedding-3-small"
    }

    COHERE = {
        "provider": "cohere",
        "api_base": "https://api.cohere.com",
        "default_model": "embed-english-v3.0"
    }

    CUSTOM = {
        "provider": "custom",
        "api_base": None,
        "default_model": None
    }


class EmbeddingFactory:
    """
    Factory class to create Embeddings instances based on the specified provider.

    Design pattern matches LLMFactory for consistency.
    Supports lazy loading and provider builder registration.
    """

    _PROVIDER_BUILDERS: Dict[
        EmbeddingProvider, 
        Callable[[Dict[str, Any], Optional[str]], Embeddings]
    ] = {}

    # Cache for frequently used embeddings
    _CACHE: Dict[str, Embeddings] = {}
    _USE_CACHE = True

    @staticmethod
    def _require_env_var(env_var: str) -> str:
        """Get required environment variable or raise error."""
        value = os.getenv(env_var)
        if not value:
            raise ValueError(f"{env_var} environment variable not set.")
        return value

    @classmethod
    def _build_huggingface(
        cls, 
        config: Dict[str, Any], 
        model_name: Optional[str]
    ) -> Embeddings:
        """Build HuggingFace embeddings (local, no API key needed)."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-huggingface not installed. "
                "Install with: pip install langchain-huggingface sentence-transformers"
            )

        model_to_use = model_name or config["default_model"]
        return HuggingFaceEmbeddings(model_name=model_to_use)

    @classmethod
    def _build_openai(
        cls,
        config: Dict[str, Any],
        model_name: Optional[str]
    ) -> Embeddings:
        """Build OpenAI embeddings."""
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-openai not installed. "
                "Install with: pip install langchain-openai"
            )

        api_key = cls._require_env_var("OPENAI_API_KEY")
        model_to_use = model_name or config["default_model"]

        return OpenAIEmbeddings(
            model=model_to_use,
            openai_api_key=api_key
        )

    @classmethod
    def _build_ollama(
        cls,
        config: Dict[str, Any],
        model_name: Optional[str]
    ) -> Embeddings:
        """Build Ollama embeddings (local self-hosted)."""
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-ollama not installed. "
                "Install with: pip install langchain-ollama"
            )

        model_to_use = model_name or config["default_model"]

        return OllamaEmbeddings(
            model=model_to_use,
            base_url=config["api_base"]
        )

    @classmethod
    def _build_ollama_cloud(
        cls,
        config: Dict[str, Any],
        model_name: Optional[str]
    ) -> Embeddings:
        """Build Ollama Cloud embeddings."""
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-ollama not installed. "
                "Install with: pip install langchain-ollama"
            )

        api_key = cls._require_env_var("OLLAMA_API_KEY")
        model_to_use = model_name or config["default_model"]

        return OllamaEmbeddings(
            model=model_to_use,
            base_url=config["api_base"],
            api_key=api_key
        )

    @classmethod
    def _build_bedrock(
        cls,
        config: Dict[str, Any],
        model_name: Optional[str]
    ) -> Embeddings:
        """Build AWS Bedrock embeddings."""
        try:
            from langchain_aws import BedrockEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-aws not installed. "
                "Install with: pip install langchain-aws"
            )

        model_to_use = model_name or config["default_model"]

        # Bedrock uses AWS credentials from environment or config
        return BedrockEmbeddings(model_id=model_to_use)

    @classmethod
    def _build_azure(
        cls,
        config: Dict[str, Any],
        model_name: Optional[str]
    ) -> Embeddings:
        """Build Azure OpenAI embeddings."""
        try:
            from langchain_openai import AzureOpenAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-openai not installed. "
                "Install with: pip install langchain-openai"
            )

        api_key = cls._require_env_var("AZURE_OPENAI_API_KEY")
        endpoint = cls._require_env_var("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        model_to_use = model_name or config["default_model"]

        return AzureOpenAIEmbeddings(
            model=model_to_use,
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )

    @classmethod
    def _build_cohere(
        cls,
        config: Dict[str, Any],
        model_name: Optional[str]
    ) -> Embeddings:
        """Build Cohere embeddings."""
        try:
            from langchain_cohere import CohereEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-cohere not installed. "
                "Install with: pip install langchain-cohere"
            )

        api_key = cls._require_env_var("COHERE_API_KEY")
        model_to_use = model_name or config["default_model"]

        return CohereEmbeddings(
            model=model_to_use,
            cohere_api_key=api_key
        )

    @classmethod
    def _get_builder(
        cls, 
        provider: EmbeddingProvider
    ) -> Optional[Callable[[Dict[str, Any], Optional[str]], Embeddings]]:
        """Get builder function for provider. Lazy loads on first use."""
        if not cls._PROVIDER_BUILDERS:
            cls._PROVIDER_BUILDERS = {
                EmbeddingProvider.HUGGINGFACE: cls._build_huggingface,
                EmbeddingProvider.OPENAI: cls._build_openai,
                EmbeddingProvider.OLLAMA: cls._build_ollama,
                EmbeddingProvider.OLLAMA_CLOUD: cls._build_ollama_cloud,
                EmbeddingProvider.BEDROCK: cls._build_bedrock,
                EmbeddingProvider.AZURE: cls._build_azure,
                EmbeddingProvider.COHERE: cls._build_cohere,
            }
        return cls._PROVIDER_BUILDERS.get(provider)

    @classmethod
    def get_embeddings(
        cls,
        provider: EmbeddingProvider = EmbeddingProvider.HUGGINGFACE,
        model_name: Optional[str] = None,
        use_cache: bool = True
    ) -> Embeddings:
        """
        Get an embeddings instance for the specified provider.

        Args:
            provider: EmbeddingProvider enum value (default: HuggingFace local)
            model_name: Optional model name override
            use_cache: Whether to cache embeddings instance (default: True)

        Returns:
            Embeddings instance ready to use

        Raises:
            ValueError: If provider not supported or env vars missing
            ImportError: If required package not installed

        Examples:
            # Local HuggingFace (default, no API keys)
            embeddings = EmbeddingFactory.get_embeddings()

            # OpenAI with custom model
            embeddings = EmbeddingFactory.get_embeddings(
                provider=EmbeddingProvider.OPENAI,
                model_name="text-embedding-3-large"
            )

            # Ollama local
            embeddings = EmbeddingFactory.get_embeddings(
                provider=EmbeddingProvider.OLLAMA
            )

            # Disable caching
            embeddings = EmbeddingFactory.get_embeddings(
                provider=EmbeddingProvider.OPENAI,
                use_cache=False
            )
        """
        # Check cache first
        cache_key = f"{provider.name}:{model_name or provider.value['default_model']}"
        if use_cache and cls._USE_CACHE and cache_key in cls._CACHE:
            return cls._CACHE[cache_key]

        # Get provider config
        provider_config = provider.value

        # Get builder
        builder = cls._get_builder(provider)
        if builder is None:
            raise ValueError(f"Unsupported embedding provider: {provider_config['provider']}")

        # Build embeddings
        embeddings = builder(provider_config, model_name)

        # Cache if enabled
        if use_cache and cls._USE_CACHE:
            cls._CACHE[cache_key] = embeddings

        return embeddings

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the embeddings cache."""
        cls._CACHE.clear()

    @classmethod
    def disable_cache(cls) -> None:
        """Disable caching globally."""
        cls._USE_CACHE = False

    @classmethod
    def enable_cache(cls) -> None:
        """Enable caching globally."""
        cls._USE_CACHE = True

    @classmethod
    def get_available_providers(cls) -> Dict[str, str]:
        """
        Get dictionary of available providers with their descriptions.

        Returns:
            Dict mapping provider name to default model
        """
        return {
            provider.name: provider.value["default_model"]
            for provider in EmbeddingProvider
            if provider != EmbeddingProvider.CUSTOM
        }

    @classmethod
    def get_provider_info(cls, provider: EmbeddingProvider) -> Dict[str, Any]:
        """
        Get detailed info about a provider.

        Args:
            provider: EmbeddingProvider enum value

        Returns:
            Dict with provider details
        """
        config = provider.value

        info = {
            "name": provider.name,
            "provider_id": config["provider"],
            "default_model": config["default_model"],
            "requires_api_key": config["api_base"] is not None or provider in [
                EmbeddingProvider.OPENAI,
                EmbeddingProvider.BEDROCK,
                EmbeddingProvider.AZURE,
                EmbeddingProvider.COHERE,
                EmbeddingProvider.OLLAMA_CLOUD,
            ],
            "local": provider in [
                EmbeddingProvider.HUGGINGFACE,
                EmbeddingProvider.OLLAMA,
            ],
        }

        if config["api_base"]:
            info["api_base"] = config["api_base"]

        return info


# ============================================================================
# Convenience Functions
# ============================================================================

def get_embeddings_by_name(provider_name: str, model_name: Optional[str] = None) -> Embeddings:
    """
    Get embeddings by provider name string.

    Args:
        provider_name: Name of provider (case-insensitive)
        model_name: Optional model override

    Returns:
        Embeddings instance

    Example:
        embeddings = get_embeddings_by_name("openai", "text-embedding-3-large")
    """
    try:
        provider = EmbeddingProvider[provider_name.upper()]
    except KeyError:
        available = ", ".join(EmbeddingFactory.get_available_providers().keys())
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Available: {available}"
        )

    return EmbeddingFactory.get_embeddings(provider=provider, model_name=model_name)


def list_embedding_providers() -> None:
    """logging.info formatted list of available embedding providers."""
    providers = EmbeddingFactory.get_available_providers()

    logging.info("\n" + "="*70)
    logging.info("AVAILABLE EMBEDDING PROVIDERS")
    logging.info("="*70)

    for provider_name, default_model in providers.items():
        info = EmbeddingFactory.get_provider_info(EmbeddingProvider[provider_name])
        local_marker = "🏠" if info["local"] else "☁️ "
        key_marker = "🔑" if info["requires_api_key"] else "✓ "

        logging.info(f"{local_marker} {key_marker} {provider_name:15} → {default_model}")

    logging.info("="*70 + "\n")

