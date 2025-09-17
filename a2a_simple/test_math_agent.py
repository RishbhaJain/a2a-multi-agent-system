#!/usr/bin/env python3
"""
Test script for the Math Agent
Run this after setting up your Gemini API key to test the agent locally
"""

import asyncio
import os
from agent_executor import MathAgent


async def test_math_agent():
    """Test the math agent with sample problems"""
    
    # Check if API key is set
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ Please set your GEMINI_API_KEY environment variable first!")
        print("   export GEMINI_API_KEY='your-api-key-here'")
        return
    
    print("🧮 Testing Math Agent with Gemini API...")
    print("=" * 50)
    
    # Create the agent
    try:
        agent = MathAgent()
        print("✅ Math agent created successfully!")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return
    
    # Test problems
    test_problems = [
        "What is 15 * 8?",
        "Solve: 3x + 5 = 17",
        "Calculate the area of a circle with radius 5",
        "What is 2^3 + 4 * 5?",
        "Find the derivative of x^2 + 3x + 2"
    ]
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n📝 Problem {i}: {problem}")
        print("-" * 30)
        
        try:
            result = await agent.invoke(problem)
            print(f"🤖 Answer:\n{result}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 50)


if __name__ == "__main__":
    asyncio.run(test_math_agent())
