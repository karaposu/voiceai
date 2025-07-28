"""
Provider Registry

Registers and manages available providers for VoiceEngine.
"""

from ..core.provider_protocol import ProviderRegistry
from .provider_adapter import ProviderAdapter
from .openai_provider import OpenAIProvider, OpenAIConfig
from .mock_provider import MockProvider, MockConfig


# Global registry instance
_global_registry = ProviderRegistry()


def get_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    return _global_registry


def register_default_providers():
    """Register all default providers."""
    # Register OpenAI provider
    openai_adapter = ProviderAdapter(OpenAIProvider, "openai")
    _global_registry.register(openai_adapter, set_as_default=True)
    
    # Register Mock provider
    mock_adapter = ProviderAdapter(MockProvider, "mock")
    _global_registry.register(mock_adapter)


# Auto-register providers on import
register_default_providers()