import os
import subprocess
import tempfile
import sys
import google.generativeai as genai


class CodeGenerationAgent:
    """Agent that generates and executes Python code using Gemini API"""
    
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            os.environ['GEMINI_API_KEY'] = "AIzaSyA6YXQspNWcJ9ra8GhG6KfzizGbIRJ6aKc"  # Temporary for testing
            api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    async def invoke(self, prompt: str) -> str:
        """Generate and execute code based on the prompt"""
        try:
            # Generate code using Gemini
            code = await self._generate_code(prompt)
            
            # Execute the code safely
            result = await self._execute_code(code)
            
            return f"Generated code:\n```python\n{code}\n```\n\nExecution result:\n{result}"
            
        except Exception as e:
            return f"Error in code generation/execution: {str(e)}"
    
    async def _generate_code(self, prompt: str) -> str:
        """Generate Python code using Gemini API"""
        system_prompt = """You are a Python code generator. Given a problem description, generate clean, efficient Python code that solves the problem.

Requirements:
1. Generate only Python code, no explanations or markdown
2. The code should be complete and executable
3. Include proper error handling where appropriate
4. Use clear variable names and comments
5. Make sure the code outputs the final result
"""
        
        full_prompt = f"{system_prompt}\n\nProblem: {prompt}"
        
        response = await self.model.generate_content_async(full_prompt)
        code = response.text.strip()
        
        # Clean up the code (remove markdown if present)
        if code.startswith('```python'):
            code = code[9:]
        if code.startswith('```'):
            code = code[3:]
        if code.endswith('```'):
            code = code[:-3]
        
        return code.strip()
    
    async def _execute_code(self, code: str) -> str:
        """Safely execute Python code and return the result"""
        try:
            # Create a temporary file to execute the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute the code in a subprocess with timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=tempfile.gettempdir()
            )
            
            # Clean up the temporary file
            os.unlink(temp_file)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if not output:
                    return "Code executed successfully but produced no output."
                return output
            else:
                error = result.stderr.strip()
                return f"Execution error:\n{error}"
                
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out (30 seconds)"
        except Exception as e:
            return f"Error executing code: {str(e)}"
