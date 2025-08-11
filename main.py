from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.config import settings
from app.routes import upload, health, embedding, workspace_agent
from app.utils.exceptions import AppException
from app.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Processing API",
    description="API for uploading and processing documents with PyMuPDF and python-docx",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(AppException)
async def app_exception_handler(request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "detail": exc.detail}
    )

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(upload.router, prefix="", tags=["upload"])
app.include_router(embedding.router, prefix="", tags=["embedding"])
app.include_router(workspace_agent.router, prefix="/api", tags=["workspace-agent"])



@app.get("/")
async def root():
    return {"message": "Document Processing API is running"}