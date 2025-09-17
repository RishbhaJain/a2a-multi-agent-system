import google.generativeai as genai
import os
from typing import Optional


class ImageRecognitionAgent:
    """Image recognition agent that identifies entities in images using Gemini Vision API"""

    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY environment variable")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    async def invoke(self, image_data: bytes, description: str = "") -> str:
        """Identify the main entity in an image"""
        try:
            # Create the prompt for simple entity identification
            prompt = "What is the main object or entity in this image? Respond with just the name of the main object (e.g., 'dog', 'cat', 'car', 'person')."
            
            # Process the image
            response = self.model.generate_content([prompt, {"mime_type": "image/png", "data": image_data}])
            
            # Extract and clean the response
            result = response.text.strip().lower()
            
            # Simple cleanup to get just the main entity name
            if result:
                # Take the first word if multiple words
                entity = result.split()[0] if result.split() else result
                return f"{entity}"
            else:
                return "I couldn't identify the main entity in this image."
                
        except Exception as e:
            return f"I encountered an error while processing the image: {str(e)}"
