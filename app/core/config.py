from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise RAG & Agent API"
    API_V1_STR: str = "/api/v1"
    
    # LLM API Keys
    OPENAI_API_KEY: str | None = None
    DEEPSEEK_API_KEY: str = "" # 增加这一行，让 pydantic 自动从 .env 读取
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = ""
    
    # Database / Vector DB Settings (For Future)
    VECTOR_DB_URL: str = "localhost"



    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

settings = Settings()
