"""
Connection State

Low-level connection state management for VoxEngine.
Tracks WebSocket connection, network metrics, and errors.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Optional
from datetime import datetime


@dataclass(frozen=True)
class ConnectionState:
    """Connection-specific state for the voice engine"""
    is_connected: bool = False
    connection_id: Optional[str] = None
    connected_at: Optional[datetime] = None
    provider: str = "openai"
    
    # Connection metrics
    latency_ms: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    
    # Error tracking
    last_error: Optional[str] = None
    error_count: int = 0
    reconnect_count: int = 0
    
    def evolve(self, **changes) -> 'ConnectionState':
        """Create a new state with specified changes"""
        return replace(self, **changes)