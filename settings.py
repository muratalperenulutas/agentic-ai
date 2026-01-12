import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')