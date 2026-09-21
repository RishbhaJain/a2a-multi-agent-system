#!/usr/bin/env python3
"""Code Execution Agent - Generates and executes Python code locally"""

import asyncio
import os
import google.generativeai as genai

from execution_sandbox import UnsafeCodeError, execute_python


class CodeExecutionAgent:
    """Agent that generates and executes Python code using Gemini API"""
    
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY environment variable")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    async def invoke(self, prompt: str) -> str:
        """Generate and execute code based on the prompt"""
        try:
            print(f"🔧 Code Execution Agent: Processing prompt...")
            
            # Generate code using Gemini
            code = await self._generate_code(prompt)
            print(f"📝 Generated code: {len(code)} characters")
            
            # Execute the code safely
            execution_result = await self._execute_code(code)
            print(f"⚡ Execution result: {execution_result}")
            
            # Extract the final numerical result
            final_result = self._extract_final_result(execution_result)
            print(f"🎯 Final result: {final_result}")
            
            return final_result
            
        except Exception as e:
            error_msg = f"Error in code generation/execution: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    async def _generate_code(self, prompt: str) -> str:
        """Generate Python code using Gemini API"""
        system_prompt = """You are an expert Python programmer. Given a problem description, generate clean, efficient Python code that solves the problem.

Requirements:
1. Generate ONLY Python code, no explanations, no markdown, no comments
2. The code must be complete and executable
3. The code must print the final numerical result
4. Use clear variable names
5. Handle edge cases appropriately
6. Make sure the code outputs the exact answer requested

Example output format:
```python
# Your code here
print(result)
```

Generate the code now:"""
        
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
        """Execute generated Python under the constrained runner."""
        try:
            result = await asyncio.to_thread(execute_python, code)

            if result.timed_out:
                return "Error: Code execution timed out (5 seconds)"
            if result.succeeded:
                output = result.stdout
                if not output:
                    return "Code executed successfully but produced no output."
                if result.output_truncated:
                    return f"{output}\n[output truncated]"
                return output
            error = result.stderr or f"process exited with code {result.returncode}"
            return f"Execution error: {error}"
        except UnsafeCodeError as exc:
            return f"Error: Generated code rejected by safety policy: {exc}"
        except Exception as e:
            return f"Error executing code: {str(e)}"
    
    def _extract_final_result(self, execution_output: str) -> str:
        """Extract the final numerical result from execution output"""
        if execution_output.startswith("Error:") or execution_output.startswith("Execution error:"):
            return execution_output
        
        # Split by lines and get the last non-empty line
        lines = [line.strip() for line in execution_output.split('\n') if line.strip()]
        if not lines:
            return "No output produced"
        
        # Return the last line (which should be the final result)
        return lines[-1]
