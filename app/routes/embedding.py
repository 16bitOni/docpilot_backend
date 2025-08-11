import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user
from app.services.chunker import MarkdownChunkerStorage
from app.services.database_service import DatabaseService
from app.utils.exceptions import AppException

logger = logging.getLogger(__name__)

router = APIRouter()

class IndexRequest(BaseModel):
    workspace_id: Optional[str] = None

class IndexResponse(BaseModel):
    message: str
    processed_files: List[Dict[str, Any]]
    total_files: int
    success_count: int
    error_count: int

class FileProcessingResult(BaseModel):
    file_id: str
    filename: str
    namespace: str
    status: str
    total_chunks: int = 0
    chunk_types: Dict[str, int] = {}
    error_message: str = None

class IndexingStatusResponse(BaseModel):
    workspace_files_count: int
    workspace_files: List[Dict[str, Any]]
    user_id: str
    workspace_id: Optional[str] = None

@router.post("/index", response_model=IndexResponse)
async def index_documents(
    request: IndexRequest,
    current_user: dict = Depends(get_current_user)
):
    """Process all files in the workspace and store them in vector database.
    Each file gets its own namespace based on the file ID."""
    
    logger.info(f"Starting document indexing process for user {current_user['id']}")
    
    processed_files = []
    success_count = 0
    error_count = 0
    
    try:
        # Initialize services
        db_service = DatabaseService()
        chunker = MarkdownChunkerStorage()
        logger.info("Services initialized successfully")
        
        # Get files from database
        if request.workspace_id:
            # Get files from specific workspace
            files = await db_service.get_workspace_files(request.workspace_id, current_user["id"])
            logger.info(f"Found {len(files)} files in workspace {request.workspace_id}")
        else:
            # Get all user files
            files = await db_service.get_user_files(current_user["id"])
            logger.info(f"Found {len(files)} files for user {current_user['id']}")
        
        if not files:
            logger.warning("No files found to index")
            return IndexResponse(
                message="No files found to index",
                processed_files=[],
                total_files=0,
                success_count=0,
                error_count=0
            )
        
        # Process each file
        for file_record in files:
            file_id = file_record["id"]
            file_name = file_record["filename"]
            filename = file_record["filename"]
            content = file_record["content"]
            
            # Create namespace from file ID (ensures uniqueness)
            namespace = f"{file_name}"
            
            logger.info(f"Processing file: {filename} (ID: {file_id}) -> namespace: {namespace}")
            
            try:
                # Process and store the file content directly
                result = chunker.process_and_store_content(
                    content=content,
                    namespace=namespace,
                    filename=filename
                )
                
                # Create successful result
                file_result = FileProcessingResult(
                    file_id=file_id,
                    filename=filename,
                    namespace=namespace,
                    status="success",
                    total_chunks=result.get("total_chunks", 0),
                    chunk_types=result.get("chunk_types", {})
                )
                processed_files.append(file_result.dict())
                success_count += 1
                
                logger.info(f"Successfully processed {filename}: {result['total_chunks']} chunks created")
                
            except Exception as e:
                error_message = f"Error processing {filename}: {str(e)}"
                logger.error(error_message)
                
                # Create error result
                file_result = FileProcessingResult(
                    file_id=file_id,
                    filename=filename,
                    namespace=namespace,
                    status="error",
                    error_message=str(e)
                )
                processed_files.append(file_result.dict())
                error_count += 1
        
        # Prepare response message
        if success_count == len(files):
            message = f"Successfully indexed all {success_count} files"
            logger.info(message)
        elif success_count > 0:
            message = f"Indexed {success_count} files successfully, {error_count} files failed"
            logger.warning(message)
        else:
            message = f"Failed to index all {error_count} files"
            logger.error(message)
        
        return IndexResponse(
            message=message,
            processed_files=processed_files,
            total_files=len(files),
            success_count=success_count,
            error_count=error_count
        )
        
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Document indexing failed")

@router.get("/index/status", response_model=IndexingStatusResponse)
async def get_indexing_status(
    workspace_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get the current status of files available for indexing"""
    
    try:
        db_service = DatabaseService()
        
        # Get files from database
        if workspace_id:
            files = await db_service.get_workspace_files(workspace_id, current_user["id"])
        else:
            files = await db_service.get_user_files(current_user["id"])
        
        # Format file information
        workspace_files = []
        for file_record in files:
            workspace_files.append({
                "file_id": file_record["id"],
                "filename": file_record["filename"],
                "file_type": file_record.get("file_type", "unknown"),
                "content_length": len(file_record.get("content", "")),
                "created_at": file_record.get("created_at"),
                "namespace": f"file_{file_record['filename']}"
            })
        
        return IndexingStatusResponse(
            workspace_files_count=len(workspace_files),
            workspace_files=workspace_files,
            user_id=current_user["id"],
            workspace_id=workspace_id
        )
        
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get indexing status")