from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX: str
    PINECONE_NAMESPACE: str = "default"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"

    # RAG
    EMBED_MODEL: str = "intfloat/e5-large-v2"
    TOP_K: int = 5
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 150
    MAX_CONTEXT_CHUNKS: int = 5

settings = Settings()
