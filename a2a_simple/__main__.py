import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from router_agent import RouterAgent
from starlette.middleware.cors import CORSMiddleware
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.utils import new_agent_text_message, new_agent_parts_message
from a2a.types import DataPart


class RouterAgentExecutor(AgentExecutor):
    """Agent executor that uses the Router Agent to handle requests"""

    def __init__(self):
        self.agent = RouterAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # Extract the question and image data from the request context
        question = ""
        image_data = None
        
        # DEBUG: Print the entire context structure
        print("=" * 80)
        print("🔍 DEBUGGING IMAGE DETECTION")
        print("=" * 80)
        print(f"Context type: {type(context)}")
        print(f"Context attributes: {[attr for attr in dir(context) if not attr.startswith('_')]}")
        
        # The message is directly on the context, not on context.request
        if hasattr(context, 'message') and context.message:
            message = context.message
            print(f"Message type: {type(message)}")
            print(f"Message attributes: {[attr for attr in dir(message) if not attr.startswith('_')]}")
            
            # Extract text and image data from message parts
            if hasattr(message, 'parts') and message.parts:
                print(f"Found {len(message.parts)} parts in message")
                
                for i, part in enumerate(message.parts):
                    print(f"\n--- Part {i} ---")
                    print(f"Part type: {type(part)}")
                    print(f"Part attributes: {[attr for attr in dir(part) if not attr.startswith('_')]}")
                    
                    if hasattr(part, 'root') and part.root:
                        root = part.root
                        print(f"Root type: {type(root)}")
                        print(f"Root attributes: {[attr for attr in dir(root) if not attr.startswith('_')]}")
                        
                        # Check for text content
                        if hasattr(root, 'text'):
                            question = root.text
                            print(f"✅ Found text: '{question}'")
                        elif hasattr(root, 'content'):
                            question = root.content
                            print(f"✅ Found content: '{question}'")
                        
                        # Check for image content
                        if hasattr(root, 'file'):
                            # This is a FilePart - extract the bytes from the file
                            file_obj = root.file
                            print(f"✅ Found file: {type(file_obj)}")
                            if hasattr(file_obj, 'bytes'):
                                image_data = file_obj.bytes
                                print(f"✅ Found file.bytes: {len(image_data)} bytes")
                            elif hasattr(file_obj, 'data'):
                                image_data = file_obj.data
                                print(f"✅ Found file.data: {len(image_data)} bytes")
                        elif hasattr(root, 'image_data'):
                            image_data = root.image_data
                            print(f"✅ Found image_data: {len(image_data)} bytes")
                        elif hasattr(root, 'data'):
                            image_data = root.data
                            print(f"✅ Found data: {len(image_data)} bytes")
                        elif hasattr(root, 'image'):
                            image_data = root.image
                            print(f"✅ Found image: {len(image_data)} bytes")
                        
                        # Print all root attributes and their types
                        print("Root attribute details:")
                        for attr in dir(root):
                            if not attr.startswith('_'):
                                try:
                                    value = getattr(root, attr)
                                    if callable(value):
                                        print(f"  {attr}: <method>")
                                    else:
                                        print(f"  {attr}: {type(value)} = {str(value)[:100]}...")
                                except:
                                    print(f"  {attr}: <unable to access>")
        
        print(f"\n📊 EXTRACTION SUMMARY:")
        print(f"Question: '{question}'")
        print(f"Image data: {len(image_data) if image_data else 'None'}")
        print("=" * 80)
        
        if not question and not image_data:
            question = "Hello! I can help with math problems, hash operations, or image recognition."
        
        result = await self.agent.invoke(question, image_data)
        
        # Check if this is an image classification response
        if image_data and self._is_classification_result(result):
            # Create classification message with DataPart
            classification_data = {
                'kind': 'classification',
                'label': result.strip()
            }
            data_part = DataPart(data=classification_data)
            message = new_agent_parts_message([data_part])
            event_queue.enqueue_event(message)
        else:
            # Use regular text message for other responses
            event_queue.enqueue_event(new_agent_text_message(result))

    def _is_classification_result(self, result: str) -> bool:
        """Check if the result is a simple classification label"""
        # Simple classification results are typically single words or short phrases
        # without punctuation or complex formatting
        result = result.strip()
        
        # Check if it's a simple word/phrase (no periods, no complex formatting)
        if len(result.split()) <= 3 and not result.endswith('.') and not result.startswith('I'):
            return True
        
        # Check for common classification patterns
        classification_patterns = [
            'dog', 'cat', 'car', 'person', 'tree', 'house', 'bird', 'fish',
            'book', 'phone', 'computer', 'chair', 'table', 'flower', 'food'
        ]
        
        return result.lower() in classification_patterns
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise Exception("Cancel not supported")


def main():
    skill = AgentSkill(
        id="multi_agent_router",
        name="Multi-Agent Router",
        description="Intelligently routes requests to specialized agents for math problems, hash operations, image recognition, and web browsing",
        tags=["math", "hash", "routing", "md5", "sha512", "calculation", "cryptography", "image", "web", "browsing", "games"],
        examples=[
            "What is 15 * 8?", 
            "Solve: 3x + 5 = 17", 
            "Calculate the area of a circle with radius 5",
            'Execute a sequence of hash operations on "hello": md5, sha512, md5',
            "Hash the word 'test' using MD5 and then SHA512",
            "Go to https://ttt.puppy9.com/ and play Tic-Tac-Toe until you win",
            "Upload an image to identify what's in it"
        ],
    )

    agent_card = AgentCard(
        name="Multi-Agent Router",
        description="An intelligent agent that routes requests to specialized sub-agents for math problems, cryptographic hash operations, image recognition, and web browsing",
        url="http://localhost:9999/",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill],
        version="1.0.0",
        capabilities=AgentCapabilities(),
    )

    request_handler = DefaultRequestHandler(
        agent_executor=RouterAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        http_handler=request_handler,
        agent_card=agent_card,
    )

    # Build the Starlette app
    app = server.build(agent_card_url="/.well-known/agent-card.json")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://ape.llm.phd", "http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    uvicorn.run(app, host="0.0.0.0", port=9999)


if __name__ == "__main__":
    main()
