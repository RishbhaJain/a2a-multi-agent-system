#!/usr/bin/env python3
"""
Test script for the Image Recognition Agent
"""

import asyncio
import os
from image_agent import ImageRecognitionAgent


async def test_image_agent():
    """Test the image recognition agent"""
    
    print("🖼️  Testing Image Recognition Agent...")
    print("=" * 50)
    
    # Create the agent (it will set the API key internally if needed)
    try:
        agent = ImageRecognitionAgent()
        print("✅ Image recognition agent created successfully!")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return
    
    # Test with a simple case (we'll create a dummy image for testing)
    print("\n📝 Testing with dummy image data...")
    print("-" * 30)
    
    # Create dummy PNG data (minimal valid PNG)
    dummy_png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82'
    
    try:
        result = await agent.invoke(dummy_png_data)
        print(f"🤖 Result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("ℹ️  Note: This test uses dummy PNG data. For real testing, use actual images.")


if __name__ == "__main__":
    asyncio.run(test_image_agent())
