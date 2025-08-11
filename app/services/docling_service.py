import logging
from pathlib import Path
import asyncio
import os

from app.utils.exceptions import AppException

logger = logging.getLogger(__name__)

class DoclingService:
    """Service for converting documents to markdown using Docling"""
    
    def __init__(self):
        try:
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()
            logger.info("Docling DocumentConverter initialized successfully")
        except ImportError as e:
            logger.error("Docling not installed. Install with: pip install docling")
            raise AppException("Docling library not available", 500, str(e))
        except Exception as e:
            logger.error(f"Failed to initialize Docling: {str(e)}")
            raise AppException("Failed to initialize document converter", 500, str(e))

    async def convert_to_markdown(self, file_path: Path) -> str:
        """Convert document to markdown format using Docling"""
        try:
            file_ext = file_path.suffix.lower()
            
            # Validate file type
            if file_ext not in ['.pdf', '.docx', '.doc', '.txt']:
                raise AppException(f"Unsupported file type: {file_ext}", 400)
            
            # For text files, read content directly
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"# {file_path.name}\n\n{content}"
            
            # Convert using Docling in a thread to avoid blocking
            def _convert_sync():
                try:
                    # Convert document using Docling
                    result = self.converter.convert(str(file_path))
                    
                    # Extract markdown content
                    markdown_content = result.document.export_to_markdown()
                    
                    return markdown_content
                    
                except Exception as e:
                    logger.error(f"Docling conversion failed for {file_path}: {str(e)}")
                    raise AppException(f"Document conversion failed: {str(e)}", 500)
            
            # Run the synchronous conversion in a thread pool
            loop = asyncio.get_event_loop()
            markdown_content = await loop.run_in_executor(None, _convert_sync)
            
            if not markdown_content or markdown_content.strip() == "":
                raise AppException("Document conversion resulted in empty content", 500)
            
            logger.info(f"Successfully converted {file_path.name} to markdown")
            return markdown_content
                
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise AppException("File not found", 404)
        except AppException:
            raise
        except Exception as e:
            logger.error(f"Document conversion failed: {str(e)}")
            raise AppException("Document conversion failed", 500, str(e))