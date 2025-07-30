"""
Identity Configuration

Defines voice assistant personalities and conversation configurations.

NOTE: Moved from voxengine to voxon for better separation of concerns.
Voice assistant identity is a high-level conversation concept.
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
            "return_transcription_text": self.return_transcription_text,
            "max_response_tokens": self.max_response_tokens,
            "input_audio_format": self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "turn_detection": {
                "enabled": self.turn_detection_enabled,
                "threshold": self.turn_detection_threshold,
                "silence_duration_ms": self.turn_detection_silence_ms
            },
            "transcription_model": self.transcription_model,
            "tool_choice": self.tool_choice
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
    prompt="""You are an engaging conversational AI. 
Listen actively, ask follow-up questions, and maintain context throughout the conversation.""",
    voice="echo",
    temperature=0.9,
    turn_detection_silence_ms=300  # Slightly longer for natural conversation
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