from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.utils import new_agent_text_message
from pydantic import BaseModel, Field
from typing import Any
import google.generativeai as genai
import os
import re


class MathAgent(BaseModel):
    """Math agent that solves basic math problems using Gemini API"""
    
    model_config = {"arbitrary_types_allowed": True}
    model: Any = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        # You'll need to set your Gemini API key here
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY environment variable")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    async def invoke(self, problem: str) -> str:
        """Solve a math problem using Gemini API"""
        try:
            # Create a focused prompt for math problems
            math_prompt = f"""
            You are a helpful math tutor. Solve this math problem step by step and provide a clear answer.
            
            Problem: {problem}
            
            Please:
            1. Show your work step by step
            2. Provide the final answer clearly
            3. Keep the explanation concise but thorough
            """
            
            response = self.model.generate_content(math_prompt)
            return response.text
            
        except Exception as e:
            return f"I encountered an error while solving the math problem: {str(e)}"


class MathAgentExecutor(AgentExecutor):

    def __init__(self):
        self.agent = MathAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # Extract the math problem from the request context
        problem = ""
        
        # The message is directly on the context, not on context.request
        if hasattr(context, 'message') and context.message:
            message = context.message
            
            # Extract text from message parts
            if hasattr(message, 'parts') and message.parts:
                for part in message.parts:
                    if hasattr(part, 'root') and part.root:
                        if hasattr(part.root, 'text'):
                            problem = part.root.text
                            break
                        elif hasattr(part.root, 'content'):
                            problem = part.root.content
                            break
        
        if not problem:
            problem = "2 + 2"  # Default problem if none provided
        
        result = await self.agent.invoke(problem)
        event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise Exception("Cancel not supported")
