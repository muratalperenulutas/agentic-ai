from llama_index.llms.ollama import Ollama
from settings import OLLAMA_URL

model_name: str = "qwen3:4b"

ollama_llm = Ollama(
    base_url=OLLAMA_URL,
    model=model_name,
    request_timeout=300.0,
    additional_kwargs={"num_ctx": 8192, "num_predict": -1},
)


ollama_embeding_llm = Ollama(
    base_url=OLLAMA_URL,
    model=model_name,
    request_timeout=300.0,
    additional_kwargs={"num_ctx": 8192, "num_predict": 512},
)