"""
Debug version of test_02 to find which test is hanging
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

print("Starting debug test 02...")

# Try importing
print("Importing VoxStream...")
from voxstream import VoxStream
print("✓ VoxStream imported")

print("Importing StreamProcessor...")
from voxstream.core.processor import StreamProcessor
print("✓ StreamProcessor imported")

print("Creating VoxStream instance...")
engine = VoxStream()
print("✓ VoxStream created")

print("Accessing processor...")
processor = engine.processor
print(f"✓ Processor accessed: {processor}")

print("\nDebug test completed successfully!")