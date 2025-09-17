#!/usr/bin/env python3
"""
Test script for the Router Agent
"""

import asyncio
from router_agent import RouterAgent


async def test_router_agent():
    """Test the router agent with different types of questions"""
    
    print("🔄 Testing Router Agent...")
    print("=" * 60)
    
    # Check if GEMINI_API_KEY is set
    import os
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  GEMINI_API_KEY not set. Testing classification only...")
        # Create a mock router for testing classification
        from router_agent import RouterAgent
        router = RouterAgent.__new__(RouterAgent)  # Create without calling __init__
        router._classify_question = RouterAgent._classify_question.__get__(router, RouterAgent)
        router._get_help_message = RouterAgent._get_help_message.__get__(router, RouterAgent)
    else:
        # Create the full router agent
        router = RouterAgent()
    
    # Test cases
    test_cases = [
        {
            "type": "Math",
            "question": "What is 15 * 8?",
            "expected": "math"
        },
        {
            "type": "Hash", 
            "question": 'Execute a sequence of hash operations on the string "hello". Perform these operations in order: 1. md5, 2. sha512',
            "expected": "hash"
        },
        {
            "type": "Math with symbols",
            "question": "Solve: 3x + 5 = 17",
            "expected": "math"
        },
        {
            "type": "Hash with keywords",
            "question": "I need to hash the word 'test' using MD5 and then SHA512",
            "expected": "hash"
        },
        {
            "type": "Unknown",
            "question": "What's the weather like today?",
            "expected": "unknown"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['type']}")
        print(f"Question: {test_case['question']}")
        print("-" * 40)
        
        try:
            # Test classification
            classification = router._classify_question(test_case['question'])
            print(f"🔍 Classified as: {classification}")
            print(f"✅ Expected: {test_case['expected']}")
            
            if classification == test_case['expected']:
                print("✅ Classification CORRECT")
            else:
                print("❌ Classification INCORRECT")
            
            # Test full routing (but skip the actual agent execution for now)
            if classification in ["math", "hash"]:
                print("🔄 Would route to appropriate agent")
            else:
                help_msg = router._get_help_message()
                print(f"ℹ️  Would show help: {help_msg[:100]}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_router_agent())
