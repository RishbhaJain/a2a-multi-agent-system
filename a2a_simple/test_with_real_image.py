#!/usr/bin/env python3
"""
Test script for the Image Recognition Agent with a real image
"""

import asyncio
import os
from PIL import Image, ImageDraw, ImageFont
from image_agent import ImageRecognitionAgent


async def test_with_real_image():
    """Test the image recognition agent with a real image"""
    
    print("🖼️  Testing Image Recognition Agent with Real Image...")
    print("=" * 60)
    
    # Create the agent
    try:
        agent = ImageRecognitionAgent()
        print("✅ Image recognition agent created successfully!")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return
    
    # Create a simple test image
    print("\n📝 Creating a simple test image...")
    print("-" * 30)
    
    try:
        # Create a simple image with text
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple rectangle (representing a "box")
        draw.rectangle([50, 30, 150, 70], outline='black', width=2)
        
        # Add text
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
        
        draw.text((60, 40), "BOX", fill='black', font=font)
        
        # Save the image
        img.save('test_image.png')
        print("✅ Created test_image.png")
        
        # Read the image data
        with open('test_image.png', 'rb') as f:
            image_data = f.read()
        
        print(f"📊 Image size: {len(image_data)} bytes")
        
        # Test the agent
        print("\n🔍 Testing image recognition...")
        result = await agent.invoke(image_data)
        print(f"🤖 Result: {result}")
        
        # Clean up
        os.remove('test_image.png')
        print("🧹 Cleaned up test image")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_with_real_image())
