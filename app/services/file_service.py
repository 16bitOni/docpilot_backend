import uuid
from pathlib import Path
from typing import Dict, Any
import logging

from fastapi import UploadFile
from app.config import settings
from app.services.document_converter import DocumentConverter
from app.services.database_service import DatabaseService
from app.utils.exceptions import AppException

logger = logging.getLogger(__name__)

class FileService:
    def __init__(self):
        self.document_converter = DocumentConverter()
        self.db_service = DatabaseService()
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(exist_ok=True)

    async def process_upload(
        self, 
        file: UploadFile, 
        user_id: str,
        workspace_id: str = None
    ) -> Dict[str, Any]:
        """Process file upload: validate, convert to markdown, save to Supabase"""
        
        # Validate file
        await self._validate_file(file)
        
        # Save file temporarily
        temp_file_path = await self._save_temp_file(file)
        
        try:
            # Convert with document converter
            markdown_content = await self.document_converter.convert_to_markdown(temp_file_path)
            
            # Update filename and file type to reflect markdown conversion
            original_name = Path(file.filename).stem  # Get filename without extension
            markdown_filename = f"{original_name}.md"
            
            # Save to Supabase database
            file_record = await self.db_service.save_converted_file(
                filename=markdown_filename,
                file_type=".md",
                content=markdown_content,
                user_id=user_id,
                workspace_id=workspace_id
            )
            
            return {
                "success": True,
                "file_id": file_record["id"],
                "filename": markdown_filename,
                "message": "File uploaded and converted to markdown successfully"
            }
            
        except Exception as e:
            logger.error(f"File processing failed: {str(e)}")
            raise AppException("File processing failed", 500, str(e))
        
        finally:
            # Clean up temp file
            if temp_file_path.exists():
                temp_file_path.unlink()

    async def _validate_file(self, file: UploadFile):
        """Validate uploaded file"""
        if file.size > settings.MAX_FILE_SIZE:
            raise AppException("File too large", 413)
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_FILE_TYPES:
            raise AppException("File type not supported", 400)

    async def _save_temp_file(self, file: UploadFile) -> Path:
        """Save uploaded file temporarily"""
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix
        temp_path = self.upload_dir / f"{file_id}{file_ext}"
        
        with open(temp_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        return temp_path

    async def get_user_files(self, user_id: str) -> list:
        """Get all converted files for a user"""
        return await self.db_service.get_user_files(user_id)

    async def get_file_by_id(self, file_id: str, user_id: str) -> Dict[str, Any]:
        """Get a specific file by ID"""
        return await self.db_service.get_file_by_id(file_id, user_id)