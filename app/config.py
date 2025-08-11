from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "Document Processing API"
    DEBUG: bool = False
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:8081", "https://docpilot-livid.vercel.app"]
    
    # Supabase
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str
    SUPABASE_JWT_SECRET: str
    
    # AI Services
    GROQ_API_KEY: str
    PINECONE_API_KEY: str
    
    # Email Services
    RESEND_API_KEY: str = ""
    
    # Brevo SMTP Settings
    BREVO_SMTP_USER: str = ""
    BREVO_SMTP_KEY: str = ""
    BREVO_FROM_EMAIL: str = ""  # Will default to BREVO_SMTP_USER if not set
    
    # Email Provider Selection (brevo, resend, auto)
    EMAIL_PROVIDER: str = "auto"
    
    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".docx", ".doc", ".txt"]
    UPLOAD_DIR: str = "uploads"
    
    # Document Conversion
    CONVERSION_CONFIG: dict = {}
    
    class Config:
        env_file = ".env"

settings = Settings()