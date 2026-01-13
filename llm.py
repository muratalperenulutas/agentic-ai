from llama_index.llms.ollama import Ollama
from settings import OLLAMA_URL

model_name: str = "qwen3:1.7b"

ollama_llm = Ollama(
    base_url=OLLAMA_URL,
    model=model_name,#modal should be bigger for generation
    request_timeout=300.0,
    additional_kwargs={"num_ctx": 8192, "num_predict": 512},
)


ollama_embeding_llm = Ollama(
    base_url=OLLAMA_URL,
    model=model_name,#modal should be smaller for embedding
    request_timeout=300.0,
    additional_kwargs={"num_ctx": 8192, "num_predict": 512},
)