# Copyright 2026 The OpenSLM Project
from pydantic_settings import BaseSettings, SettingsConfigDict

class MiraiSettings(BaseSettings):
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"
    
    # AI Settings
    TOKENIZER_PATH: str = "models/ecommerce_tokenizer.model"
    BASE_MODEL_ID: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = MiraiSettings()
