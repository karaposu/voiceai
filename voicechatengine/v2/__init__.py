# VoiceEngine V2 - Provider abstraction implementation

from .providers import (
    # Base classes
    BaseProvider,
    ProviderConfig,
    ProviderEvent,
    AudioFormat,
    MessageType,
    ConnectionState,
    
    # Errors
    ProviderError,
    ConnectionError,
    MessageError,
    AudioError,
    
    # Providers
    OpenAIProvider,
    OpenAIConfig,
    MockProvider,
    MockConfig
)

__version__ = "2.0.0"

__all__ = [
    # Base classes
    'BaseProvider',
    'ProviderConfig',
    'ProviderEvent',
    'AudioFormat',
    'MessageType',
    'ConnectionState',
    
    # Errors
    'ProviderError',
    'ConnectionError',
    'MessageError',
    'AudioError',
    
    # Providers
    'OpenAIProvider',
    'OpenAIConfig',
    'MockProvider',
    'MockConfig'
]