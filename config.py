import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Other configuration settings
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-opus-20240229")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4-0125-preview")
