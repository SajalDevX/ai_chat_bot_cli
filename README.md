# AI Chat Bot CLI

A command-line chatbot powered by Google's Gemini API with streaming responses and a beautiful terminal interface.

## Features

- Interactive chat with Google Gemini AI models
- Real-time streaming responses
- Conversation history (AI remembers context)
- Beautiful terminal UI with Rich
- Easy configuration via `.env` file

## Prerequisites

- Python 3.12 or higher
- A Google Gemini API key ([Get one here](https://aistudio.google.com/apikey))

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chat_bot_cli
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv pip install -e .

   # Or using pip
   pip install -e .
   ```

4. **Configure your API key**

   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-2.0-flash
   MAX_TOKENS=1000
   TEMPERATURE=0.7
   ```

## Usage

**Run the chatbot:**
```bash
# Using the CLI command
ai-chat-bot

# Or using Python module
python -m ai_chat_bot.main

# Or with uv
uv run python -m ai_chat_bot.main
```

**Example session:**
```
╭─────────────────────────────────────────────────────────╮
│ AI Chatbot                                              │
│ Type your message and press Enter.                     │
│ Commands: /help, /clear, /quit                         │
╰─────────────────────────────────────────────────────────╯

✅ Connected to Gemini API!
ℹ️  Model: gemini-2.0-flash
ℹ️  Streaming enabled

You: Hello! What can you help me with?

🤖 Assistant:
Hello! I can help you with a wide range of tasks...
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help information |
| `/clear` | Clear conversation history |
| `/quit` | Exit the chatbot |

## Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Your Google Gemini API key | Required |
| `GEMINI_MODEL` | Gemini model to use | `gemini-2.0-flash` |
| `MAX_TOKENS` | Maximum tokens in response | `1000` |
| `TEMPERATURE` | Response creativity (0.0-2.0) | `0.7` |
| `TIMEOUT` | API request timeout (seconds) | `30.0` |

### Available Models

- `gemini-2.0-flash` - Fast and efficient (recommended)
- `gemini-2.5-flash` - Latest flash model
- `gemini-2.5-pro` - Most capable model

## Project Structure

```
chat_bot_cli/
├── src/
│   └── ai_chat_bot/
│       ├── __init__.py
│       ├── main.py           # Entry point and ChatBot class
│       ├── config.py         # Settings configuration
│       ├── clients/
│       │   ├── __init__.py
│       │   └── gemini.py     # Gemini API client
│       ├── models/
│       │   ├── __init__.py
│       │   └── messages.py   # Message and Conversation models
│       ├── ui/
│       │   ├── __init__.py
│       │   └── display.py    # Terminal display utilities
│       └── utils/
│           ├── __init__.py
│           └── exceptions.py # Custom exceptions
├── .env                      # Environment variables (create this)
├── pyproject.toml
└── README.md
```

## Troubleshooting

### "Rate limit exceeded" error
- Check your API quota at https://ai.dev/usage
- Try using `gemini-2.0-flash` model (more generous limits)
- Wait a few minutes and try again

### "API key not valid" error
- Verify your API key at https://aistudio.google.com/apikey
- Make sure the `.env` file is in the project root
- Check that the key is correctly copied (no extra spaces)

### "Model not found" error
- Update to a supported model (e.g., `gemini-2.0-flash`)
- Check available models at https://ai.google.dev/gemini-api/docs/models

## License

MIT License