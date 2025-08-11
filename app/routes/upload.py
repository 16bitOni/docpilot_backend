from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Optional
import logging

from app.dependencies import get_current_user
from app.services.file_service import FileService
from app.models.response import FileUploadResponse
from app.utils.exceptions import AppException

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    workspace_id: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """Upload and convert document file to markdown, save to Supabase"""
    try:
        file_service = FileService()
        
        result = await file_service.process_upload(
            file=file,
            user_id=current_user["id"],
            workspace_id=workspace_id
        )
        
        return FileUploadResponse(**result)
        
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="File upload failed")
@router.get("/files")
async def get_user_files(
    current_user: dict = Depends(get_current_user)
):
    """Get all converted files for the current user"""
    try:
        file_service = FileService()
        files = await file_service.get_user_files(current_user["id"])
        
        return {
            "success": True,
            "files": files,
            "count": len(files)
        }
        
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to get files: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve files")

@router.get("/files/{file_id}")
async def get_file(
    file_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific converted file by ID"""
    try:
        file_service = FileService()
        file_data = await file_service.get_file_by_id(file_id, current_user["id"])
        
        return {
            "success": True,
            "file": file_data
        }
        
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to get file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve file")