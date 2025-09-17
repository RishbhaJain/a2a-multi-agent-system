from typing import Optional
from agent_executor import MathAgent
from hash_agent import HashAgent
from image_agent import ImageRecognitionAgent
from web_agent import WebBrowsingAgent
from memory_agent import MemoryAgent
from code_execution_agent import CodeExecutionAgent


class RouterAgent:
    """Router agent that directs requests to appropriate specialized agents"""

    def __init__(self):
        self.math_agent = MathAgent()
        self.hash_agent = HashAgent()
        self.image_agent = ImageRecognitionAgent()
        self.web_agent = WebBrowsingAgent()
        self.memory_agent = MemoryAgent()
        self.code_agent = CodeExecutionAgent()

    async def invoke(self, question: str, image_data: Optional[bytes] = None) -> str:
        """Route the question to the appropriate agent"""
        try:
            # If image data is provided, route to image recognition agent
            if image_data:
                print("🖼️ Routing to Image Recognition Agent")
                return await self.image_agent.invoke(image_data, question)
            
            # Determine the question type for text-only requests
            question_type = self._classify_question(question)
            
            if question_type == "math":
                print("🔢 Routing to Math Agent")
                return await self.math_agent.invoke(question)
            elif question_type == "hash":
                print("🔐 Routing to Hash Agent")
                return await self.hash_agent.invoke(question)
            elif question_type == "web":
                print("🌐 Routing to Web Browsing Agent")
                return await self.web_agent.invoke(question)
            elif question_type == "memory":
                print("🧠 Routing to Memory Agent")
                return self._handle_memory_request(question)
            elif question_type == "code":
                print("💻 Routing to Code Execution Agent")
                return await self.code_agent.invoke(question)
            else:
                return self._get_help_message()
                
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}"
    
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
        """Classify the question as math, hash, web, or unknown"""
        question_lower = question.lower()
        
        # Hash operation keywords
        hash_keywords = [
            "hash", "md5", "sha512", "sha", "execute", "sequence", 
            "operations", "cryptographic", "digest"
        ]
        
        # Math operation keywords
        math_keywords = [
            "solve", "calculate", "what is", "math", "arithmetic", 
            "algebra", "equation", "derivative", "integral", "area", 
            "perimeter", "volume", "plus", "minus", "times", "divided"
        ]
        
        # Web browsing keywords
        web_keywords = [
            "go to", "visit", "browse", "website", "play", "game", 
            "tic-tac-toe", "ttt.puppy9.com", "secret number", "congratulation",
            "win", "browser", "web", "url", "http", "https"
        ]
        
        # Memory keywords
        memory_keywords = [
            "remember", "store", "save", "recall", "retrieve", "memory",
            "previously", "earlier", "before", "past", "stored", "saved",
            "what was", "paired with", "check your memory", "tell me"
        ]
        
        # Code execution keywords
        code_keywords = [
            "write a program", "program that", "compute", "algorithm", "prime numbers",
            "sum of squares", "modulo", "programming", "code", "execute", "generate code",
            "write code", "implement", "calculate", "complex", "mathematical problem",
            "step by step", "numerical result", "final result", "output the result"
        ]
        
        # Check for hash operations
        hash_score = sum(1 for keyword in hash_keywords if keyword in question_lower)
        
        # Check for math operations
        math_score = sum(1 for keyword in math_keywords if keyword in question_lower)
        
        # Check for web browsing
        web_score = sum(1 for keyword in web_keywords if keyword in question_lower)
        
        # Check for memory operations
        memory_score = sum(1 for keyword in memory_keywords if keyword in question_lower)
        
        # Check for code execution operations
        code_score = sum(1 for keyword in code_keywords if keyword in question_lower)
        
        # Also check for mathematical symbols
        math_symbols = ["+", "-", "*", "/", "=", "^", "√", "π", "x", "y"]
        math_symbol_count = sum(1 for symbol in math_symbols if symbol in question)
        
        if math_symbol_count > 0:
            math_score += math_symbol_count
        
        # Determine the classification (memory has highest priority, then code)
        if memory_score > 0:
            return "memory"
        elif code_score > 0:
            return "code"
        elif web_score > 0:
            return "web"
        elif hash_score > math_score and hash_score > 0:
            return "hash"
        elif math_score > 0:
            return "math"
        else:
            return "unknown"

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
