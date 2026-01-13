from llama_index.embeddings.ollama import OllamaEmbedding
from settings import OLLAMA_URL


ollama_embedding = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=OLLAMA_URL,
    request_timeout=300.0,
    ollama_additional_kwargs={"mirostat": 0},
)
