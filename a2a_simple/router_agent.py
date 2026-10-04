from collections.abc import Awaitable, Callable
from typing import Optional

from agent_resilience import CircuitBreaker, invoke_with_resilience
from agent_executor import MathAgent
from hash_agent import HashAgent
from image_agent import ImageRecognitionAgent
from web_agent import WebBrowsingAgent
from memory_agent import MemoryAgent
from code_execution_agent import CodeExecutionAgent
from routing_policy import classify_question
from telemetry import AgentTelemetry


class RouterAgent:
    """Router agent that directs requests to appropriate specialized agents"""

    def __init__(
        self,
        telemetry: AgentTelemetry | None = None,
        *,
        agent_timeout_seconds: float = 30.0,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        if agent_timeout_seconds <= 0:
            raise ValueError("agent_timeout_seconds must be positive")
        self.math_agent = MathAgent()
        self.hash_agent = HashAgent()
        self.image_agent = ImageRecognitionAgent()
        self.web_agent = WebBrowsingAgent()
        self.memory_agent = MemoryAgent()
        self.code_agent = CodeExecutionAgent()
        self.telemetry = telemetry or AgentTelemetry()
        self.agent_timeout_seconds = agent_timeout_seconds
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

    async def invoke(
        self,
        question: str,
        image_data: Optional[bytes] = None,
        request_id: str | None = None,
    ) -> str:
        """Route the question to the appropriate agent"""
        span = self.telemetry.start(
            request_id=request_id,
            input_text_chars=len(question),
            input_image_bytes=len(image_data) if image_data else 0,
        )
        try:
            # If image data is provided, route to image recognition agent
            if image_data:
                route = "image"
                span.set_route(route)
                result = await self._invoke_agent(
                    route, lambda: self.image_agent.invoke(image_data, question)
                )
            else:
                route = self._classify_question(question)
                if route not in {"math", "hash", "web", "memory", "code"}:
                    route = "unsupported"
                span.set_route(route)
                if route == "math":
                    result = await self._invoke_agent(
                        route, lambda: self.math_agent.invoke(question)
                    )
                elif route == "hash":
                    result = await self._invoke_agent(
                        route, lambda: self.hash_agent.invoke(question)
                    )
                elif route == "web":
                    result = await self._invoke_agent(
                        route, lambda: self.web_agent.invoke(question)
                    )
                elif route == "memory":
                    result = self._handle_memory_request(question)
                elif route == "code":
                    result = await self._invoke_agent(
                        route, lambda: self.code_agent.invoke(question)
                    )
                else:
                    result = self._get_help_message()

            span.finish()
            return result
        except Exception as error:
            span.fail(error)
            return (
                "The selected agent is temporarily unavailable. "
                "Please try the request again."
            )

    async def _invoke_agent(
        self,
        route: str,
        operation: Callable[[], Awaitable[str]],
    ) -> str:
        return await invoke_with_resilience(
            route,
            operation,
            timeout_seconds=self.agent_timeout_seconds,
            circuit_breaker=self.circuit_breaker,
        )

    def _handle_memory_request(self, question: str) -> str:
        """Handle memory-related requests"""
        try:
            question_lower = question.lower()
            
            # Check if this is a store request
            if any(word in question_lower for word in ["remember", "store", "save"]):
                return self._parse_store_request(question)
            
            # Check if this is a retrieve request
            elif any(word in question_lower for word in ["recall", "retrieve", "what was", "previously", "check your memory", "tell me"]):
                return self._parse_retrieve_request(question)
            
            else:
                return "I can help you store or retrieve information from memory. Please specify what you'd like to remember or recall."
                
        except Exception as e:
            return f"Error handling memory request: {str(e)}"
    
    def _parse_store_request(self, question: str) -> str:
        """Parse and handle store requests"""
        try:
            # Look for number pairs in the format "X and Y"
            import re
            
            # Pattern to find "number and number"
            pattern = r'(\d+)\s+and\s+(\d+)'
            match = re.search(pattern, question)
            
            if match:
                key = match.group(1)
                value = match.group(2)
                return self.memory_agent.store(key, value)
            else:
                return "I couldn't find a number pair to store. Please use the format 'X and Y'."
                
        except Exception as e:
            return f"Error parsing store request: {str(e)}"
    
    def _parse_retrieve_request(self, question: str) -> str:
        """Parse and handle retrieve requests"""
        try:
            # Look for a number to search for
            import re
            
            # Find numbers in the question
            numbers = re.findall(r'\d+', question)
            
            if numbers:
                # Use the first number found as search term
                search_term = numbers[0]
                return self.memory_agent.retrieve(search_term)
            else:
                return "I couldn't find a number to search for. Please specify which number you're looking for."
                
        except Exception as e:
            return f"Error parsing retrieve request: {str(e)}"

    def _classify_question(self, question: str) -> str:
        """Classify a request with the versioned, evaluated routing policy."""
        return classify_question(question).tool

    def _get_help_message(self) -> str:
        """Return a help message when question type is unknown"""
        return """I can help you with six types of problems:

🔢 **Math Problems**: 
   - Basic arithmetic (2 + 2, 15 * 8)
   - Algebra (solve: 3x + 5 = 17)
   - Calculus (derivatives, integrals)
   - Geometry (area, perimeter calculations)

🔐 **Hash Operations**: 
   - Execute sequences of MD5 and SHA512 operations
   - Example: "Execute a sequence of hash operations on 'hello': md5, sha512, md5"

🖼️ **Image Recognition**: 
   - Upload a PNG image to identify the main entity
   - Example: Upload an image of a dog → "dog"

🌐 **Web Browsing**: 
   - Browse websites and interact with web pages
   - Play games and extract information
   - Example: "Go to https://ttt.puppy9.com/ and play Tic-Tac-Toe until you win"

🧠 **Memory Operations**: 
   - Store information for future reference
   - Retrieve previously stored information
   - Example: "Remember 123 and 456" then "What was paired with 123?"

💻 **Code Execution**: 
   - Solve complex programming problems
   - Generate and execute Python code
   - Mathematical algorithms and computations
   - Example: "Write a program that computes the sum of squares of all prime numbers from 1 to n, modulo 1000"

Please rephrase your question with more specific details about what you'd like me to help you with."""
