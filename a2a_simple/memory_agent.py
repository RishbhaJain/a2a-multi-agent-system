import json
import os
from datetime import datetime
from typing import Optional, Dict, Any


class MemoryAgent:
    """Agent that stores and retrieves information in persistent memory"""
    
    def __init__(self, memory_file: str = "memory_data.json"):
        self.memory_file = memory_file
        self.memory_data = self._load_memory()
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory data from JSON file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            else:
                return {"entries": []}
        except Exception as e:
            print(f"Error loading memory: {e}")
            return {"entries": []}
    
    def _save_memory(self):
        """Save memory data to JSON file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory_data, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def store(self, key: str, value: str) -> str:
        """Store a key-value pair in memory"""
        try:
            # Create new entry
            entry = {
                "id": str(len(self.memory_data["entries"]) + 1),
                "key": key,
                "value": value,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to memory
            self.memory_data["entries"].append(entry)
            self._save_memory()
            
            return f"Stored: {key} = {value}"
            
        except Exception as e:
            return f"Error storing memory: {str(e)}"
    
    def retrieve(self, search_term: str) -> str:
        """Retrieve information from memory"""
        try:
            # Search through entries
            for entry in self.memory_data["entries"]:
                if search_term in entry["key"] or search_term in entry["value"]:
                    return f"Found: {entry['key']} = {entry['value']}"
            
            return f"Not found: {search_term}"
            
        except Exception as e:
            return f"Error retrieving memory: {str(e)}"
    
    def clear(self) -> str:
        """Clear all memory"""
        try:
            self.memory_data = {"entries": []}
            self._save_memory()
            return "Memory cleared successfully"
        except Exception as e:
            return f"Error clearing memory: {str(e)}"
