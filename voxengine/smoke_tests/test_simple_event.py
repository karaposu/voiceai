"""
Simple test to verify event system works
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
from ..voice_engine import VoiceEngine
from ..events import EventType

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

async def main():
    print("Testing VoiceEngine Event System...")
    
    engine = VoiceEngine(api_key=API_KEY, mode="fast")
    
    # Track events
    events_received = []
    
    # Use callback (should work)
    engine.on_text_response = lambda text: print(f"[CALLBACK] Got text: {text}")
    
    # Use event system
    engine.events.on(EventType.TEXT_OUTPUT, 
                    lambda event: events_received.append(event.text) or print(f"[EVENT] Got text: {event.text}"))
    
    # Add debug handler for all events
    engine.events.on("*", lambda event: print(f"[DEBUG] Event: {event.type}"))
    
    print("Connecting...")
    await engine.connect()
    
    print("Sending text...")
    await engine.send_text("Say hello in 3 words")
    
    print("Waiting for response...")
    await asyncio.sleep(5)
    
    print(f"\nEvents received: {len(events_received)}")
    if events_received:
        print(f"Text: {events_received[0]}")
    
    await engine.disconnect()
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())