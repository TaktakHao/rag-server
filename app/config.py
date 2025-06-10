from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    # API设置
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Enterprise RAG API"
    
    # 模型设置
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    EMBEDDING_MODEL_PATH: str = "Your Embedding Model Path"
    
    # 向量数据库设置
    VECTOR_DB_DIR: Path = Path("./data/vector_db")
    KNOWLEDGE_BASE_DIR: Path = Path("./data/knowledge_base")
    
    # 文本分块设置
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_RETRIEVAL_DOCS: int = 5
    
    class Config:
        case_sensitive = True

settings = Settings()

# 创建必要的目录
settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
settings.KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True) 