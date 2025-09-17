import hashlib
import re
from typing import List


class HashAgent:
    """Hash agent that executes sequences of MD5 and SHA512 operations"""

    def __init__(self):
        pass

    async def invoke(self, operation_description: str) -> str:
        """Execute a sequence of hash operations"""
        try:
            # Extract the input string and operations from the description
            input_string, operations = self._parse_operation(operation_description)
            
            if not input_string or not operations:
                return "I couldn't parse the hash operation. Please provide a clear description with the input string and operation sequence."
            
            # Execute the hash sequence
            result = self._execute_hash_sequence(input_string, operations)
            
            return f"Hash sequence completed successfully!\n\nInput: '{input_string}'\nOperations: {', '.join(operations)}\nFinal result: {result}"
            
        except Exception as e:
            return f"I encountered an error while executing the hash operations: {str(e)}"

    def _parse_operation(self, description: str) -> tuple[str, List[str]]:
        """Parse the operation description to extract input string and operations"""
        # Look for the input string (usually in quotes)
        string_match = re.search(r'"([^"]*)"', description)
        input_string = string_match.group(1) if string_match else ""
        
        # Look for operation sequence
        operations = []
        description_lower = description.lower()
        
        # Find all occurrences of operations in order
        i = 0
        while i < len(description_lower):
            if description_lower[i:i+3] == "md5":
                operations.append("md5")
                i += 3
            elif description_lower[i:i+6] == "sha512":
                operations.append("sha512")
                i += 6
            else:
                i += 1
        
        return input_string, operations

    def _execute_hash_sequence(self, input_string: str, operations: List[str]) -> str:
        """Execute the hash sequence step by step"""
        current_value = input_string
        
        for i, operation in enumerate(operations):
            if operation.lower() == "md5":
                current_value = hashlib.md5(current_value.encode()).hexdigest()
            elif operation.lower() == "sha512":
                current_value = hashlib.sha512(current_value.encode()).hexdigest()
            else:
                raise ValueError(f"Unknown hash operation: {operation}")
        
        return current_value
