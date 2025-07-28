"""
Configuration file for RealtimeVoiceAPI

This file contains Identity dataclass and predefined identities
for easy customization without modifying core code.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Identity:
    """Represents a voice assistant identity with all its configuration"""
    
    name: str
    prompt: str
    voice: str = "alloy"
    temperature: float = 0.8
    return_transcription_text: bool = True
    modalities: List[str] = field(default_factory=lambda: ["text", "audio"])
    max_response_tokens: int = 4096
    
    # Audio settings
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    
    # Turn detection
    turn_detection_enabled: bool = True
    turn_detection_threshold: float = 0.5
    turn_detection_silence_ms: int = 200
    
    # Optional settings
    transcription_model: Optional[str] = "whisper-1"
    tool_choice: str = "auto"
    
    def to_session_config_dict(self) -> dict:
        """Convert Identity to dictionary format expected by SessionConfig"""
        return {
            "instructions": self.prompt,
            "voice": self.voice,
            "temperature": self.temperature,
            "modalities": self.modalities,
            "input_audio_format": self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "max_response_output_tokens": self.max_response_tokens,
            "tool_choice": self.tool_choice,
            "turn_detection": {
                "type": "server_vad" if self.turn_detection_enabled else None,
                "threshold": self.turn_detection_threshold,
                "silence_duration_ms": self.turn_detection_silence_ms,
            } if self.turn_detection_enabled else None,
            "input_audio_transcription": {
                "model": self.transcription_model
            } if self.transcription_model else None,
        }


# Predefined identities
DEFAULT_ASSISTANT = Identity(
    name="default",
    prompt="You are a helpful assistant.",
    voice="alloy",
    temperature=0.8
)

VOICE_ASSISTANT = Identity(
    name="voice_assistant",
    prompt="You are a helpful voice assistant. Be concise and friendly.",
    voice="alloy",
    temperature=0.8
)

TRANSCRIPTION_SERVICE = Identity(
    name="transcription",
    prompt="Transcribe audio accurately. Do not add any commentary.",
    voice="alloy",
    temperature=0.3,
    modalities=["text"],  # Text only for transcription
    return_transcription_text=True,
    turn_detection_enabled=False
)

CONVERSATIONAL_AI = Identity(
    name="conversational",
    prompt="You are a conversational AI. Engage naturally and ask follow-up questions when appropriate.",
    voice="alloy",
    temperature=0.9,
    turn_detection_silence_ms=300  # Longer pause for natural conversation
)

CUSTOMER_SERVICE = Identity(
    name="customer_service",
    prompt="""You are a professional customer service representative. 
Be polite, helpful, and solution-oriented. Ask clarifying questions when needed.""",
    voice="shimmer",  # Professional sounding voice
    temperature=0.7,
    max_response_tokens=2048  # Shorter responses for efficiency
)

AUDIO_ONLY_ASSISTANT = Identity(
    name="audio_only",
    prompt="Communicate only through speech. Be clear and expressive.",
    voice="nova",
    temperature=0.8,
    modalities=["audio"],  # Audio only
    return_transcription_text=False
)

# Dictionary for easy access to predefined identities
IDENTITIES = {
    "default": DEFAULT_ASSISTANT,
    "voice_assistant": VOICE_ASSISTANT,
    "transcription": TRANSCRIPTION_SERVICE,
    "conversational": CONVERSATIONAL_AI,
    "customer_service": CUSTOMER_SERVICE,
    "audio_only": AUDIO_ONLY_ASSISTANT,
}