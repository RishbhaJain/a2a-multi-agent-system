# Multi-Agent Router System

A sophisticated Agent-to-Agent (A2A) system that intelligently routes requests to specialized sub-agents for math problems, cryptographic hash operations, image recognition, web browsing, code execution, and memory management. Built using the A2A SDK and powered by Google's Gemini API.

## 🌟 Features

### 🔢 **Math Agent**
- Solves arithmetic, algebra, and calculus problems
- Uses Gemini Pro API for intelligent problem-solving
- Handles complex mathematical expressions and equations

### 🔐 **Hash Agent**
- Executes sequences of MD5 and SHA512 operations
- Supports complex cryptographic hash chains
- Processes text input and returns hash results

### 🖼️ **Image Recognition Agent**
- Identifies entities in PNG images using Gemini Vision API
- Returns structured classification format: `{kind: 'classification', label: 'dog'}`
- Handles image uploads through A2A protocol
- Supports both simple entity names and complex image analysis

### 🌐 **Web Browsing Agent**
- Navigates websites and interacts with web pages
- Plays games and extracts information
- Uses Selenium for browser automation
- Can solve complex web-based challenges (e.g., Tic-Tac-Toe games)

### 💻 **Code Execution Agent**
- Generates Python code using Gemini AI
- Validates generated code with an AST policy before execution
- Runs code in an isolated Python subprocess with CPU, memory, file, and output limits
- Solves complex programming and mathematical problems
- Returns only the final numerical result
- Handles prime number calculations, algorithms, and mathematical computations

### 🧠 **Memory Agent**
- Stores key-value pairs for future reference
- Retrieves previously stored information across sessions
- Persistent memory using JSON file storage
- Supports complex memory operations and queries

### 🧠 **Smart Router**
- Automatically classifies incoming requests
- Routes to appropriate specialized agents
- Provides helpful guidance for unsupported requests

## 🚀 Quick Start

### Prerequisites

1. **uv:** Python package management tool
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **Python 3.11+** - Required for the A2A SDK

3. **Gemini API Key** - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

4. **Chrome Browser** - Required for web browsing agent (auto-installed via webdriver-manager)

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd agent2agent/a2a_simple
   ```

2. **Set up virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate  # or .venv2/bin/activate if using .venv2
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

4. **Set your Gemini API key:**
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
   
   **⚠️ Security Note:** Never commit your API key to version control. The code will throw an error if the API key is not set as an environment variable.

### Running the System

Start the multi-agent server:
```bash
uv run .
```

The system will be available at `http://localhost:9999`

## 📋 Usage Examples

### Math Problems
```
"What is 15 * 8?"
"Solve: 3x + 5 = 17"
"Calculate the area of a circle with radius 5"
"Find the derivative of x^2 + 3x + 1"
```

### Hash Operations
```
"Execute a sequence of hash operations on 'hello': md5, sha512, md5"
"Hash the word 'test' using MD5 and then SHA512"
"Apply MD5 to 'password' then SHA512 to the result"
```

### Image Recognition
Upload a PNG image and ask:
```
"What's in this image?"
"Identify the main object in this picture"
```

### Web Browsing
```
"Go to https://ttt.puppy9.com/ and play Tic-Tac-Toe until you win"
"Visit https://example.com and tell me the page title"
"Browse to a website and extract specific information"
```

### Code Execution
```
"Write a program that computes the sum of squares of all prime numbers from 1 to n, modulo 1000"
"Calculate the factorial of 10 using a program"
"Find all prime numbers between 1 and 100 and sum them"
"Write a program to solve the Fibonacci sequence up to the 20th term"
```

### Memory Operations
```
"Remember this number pair for future reference: 123 and 456"
"Store the value 789 with key 'test_key'"
"What was paired with 123?"
"Retrieve the value for key 'test_key'"
"Check your memory for any stored information"
```

## 🏗️ Architecture

```
┌─────────────────┐
│   A2A Client    │
└─────────┬───────┘
          │
┌─────────▼───────┐
│  Router Agent   │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
┌───▼───┐   ┌───▼───┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Math  │   │ Hash  │   │ Image   │   │  Web    │   │  Code   │   │ Memory  │
│ Agent │   │ Agent │   │ Agent   │   │ Agent   │   │ Agent   │   │ Agent   │
└───────┘   └───────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

### Components

- **RouterAgent**: Main orchestrator that classifies requests and routes to appropriate agents
- **MathAgent**: Handles mathematical problem-solving using Gemini Pro
- **HashAgent**: Executes cryptographic hash operations (MD5, SHA512)
- **ImageRecognitionAgent**: Processes images using Gemini Vision API
- **WebBrowsingAgent**: Controls web browsers using Selenium for automation
- **CodeExecutionAgent**: Generates and executes Python code for complex problems
- **MemoryAgent**: Manages persistent storage and retrieval of key-value pairs

## 🔧 Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)

### Dependencies

The system uses the following key dependencies:
- `a2a-sdk>=0.2.5`: Agent-to-Agent communication
- `google-generativeai>=0.8.0`: Gemini API integration
- `selenium>=4.15.0`: Web browser automation
- `webdriver-manager>=4.0.0`: Automatic browser driver management
- `pillow>=11.3.0`: Image processing
- `beautifulsoup4>=4.12.0`: HTML parsing
- `requests>=2.31.0`: HTTP requests

## 🧪 Testing

### Test Individual Agents

```bash
# Test math agent
python test_math_agent.py

# Test hash agent  
python test_hash_agent.py

# Test image agent
python test_image_agent.py

# Test router classification
python test_router_agent.py
```

### Test Full System

```bash
# Test with A2A client
python test_client.py

# Test integrated system
python test_integrated_system.py
```

## 🌐 API Endpoints

- `GET /.well-known/agent-card.json`: Agent capabilities and metadata
- `POST /`: Main A2A message endpoint

## 🔒 Security

### API Key Management
- **Never commit API keys to version control**
- Always use environment variables for sensitive data
- The system will throw an error if `GEMINI_API_KEY` is not set
- Consider using `.env` files for local development (add to `.gitignore`)

### Best Practices
- Rotate your API keys regularly
- Use least-privilege access for API keys
- Monitor API usage for unusual activity
- Keep dependencies updated for security patches

### Constrained Code Execution

Generated Python is checked before execution. The policy blocks filesystem,
process, network, dynamic-evaluation, and dunder-based escape primitives while
allowing a small set of computation-focused standard-library modules. Accepted
programs run with Python isolation flags, a five-second timeout, memory and file
limits, and bounded captured output. The policy has credential-free regression
tests in CI.

This is defense in depth for a portfolio system, not a hardened isolation
boundary. Adversarial untrusted code should run in a network-denied container or
microVM with an unprivileged user and a read-only filesystem.

## 🔍 Troubleshooting

### Common Issues

1. **"uv not found"**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **"GEMINI_API_KEY not set"**
   ```bash
   export GEMINI_API_KEY="your-actual-api-key"
   ```

3. **Web browsing not working**
   - Ensure Chrome browser is installed
   - Check internet connection
   - Verify the target website is accessible

4. **Image recognition failing**
   - Ensure image is in PNG format
   - Check image file size (not too large)
   - Verify Gemini API key has vision permissions

5. **Code execution errors**
   - Check that Python is properly installed
   - Verify code execution timeout (30 seconds)
   - Ensure generated code is syntactically correct

6. **Memory operations not working**
   - Check file permissions for `memory_data.json`
   - Verify memory file is not corrupted
   - Ensure proper JSON format in memory storage

### Debug Mode

Enable debug logging by adding print statements in the agent files or using Python's logging module.

## 📁 Project Structure

```
a2a_simple/
├── __main__.py              # Main server and A2A setup
├── router_agent.py          # Smart request router
├── agent_executor.py        # Math agent implementation
├── hash_agent.py           # Hash operations agent
├── image_agent.py          # Image recognition agent
├── web_agent.py            # Web browsing agent
├── code_execution_agent.py # Code generation and execution agent
├── memory_agent.py         # Memory storage and retrieval agent
├── code_agent.py           # Legacy code generation agent (commented out)
├── test_*.py               # Test files for individual agents
├── pyproject.toml          # Project dependencies
└── README.md               # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is part of the A2A (Agent-to-Agent) ecosystem. Please refer to the A2A SDK license for usage terms.

## 🔗 References

- [A2A Python SDK](https://github.com/google/a2a-python)
- [Google AI Studio](https://makersuite.google.com/app/apikey)
- [Selenium Documentation](https://selenium-python.readthedocs.io/)
- [Gemini API Documentation](https://ai.google.dev/docs)

## 🎯 Future Enhancements

- [ ] Add more specialized agents (file processing, database queries)
- [ ] Implement agent-to-agent communication
- [ ] Add support for more image formats
- [ ] Enhance web browsing capabilities
- [ ] Add support for more programming languages in code execution
- [ ] Implement conversation memory across sessions
- [ ] Add authentication and security features
- [ ] Add support for more complex data structures in memory agent
- [ ] Implement code execution sandboxing improvements

---

**Built with ❤️ using the A2A SDK and Google Gemini AI**
