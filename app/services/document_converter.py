import logging
from pathlib import Path
import asyncio
import os

from app.utils.exceptions import AppException

logger = logging.getLogger(__name__)

class DocumentConverter:
    """Service for converting documents to markdown using PyMuPDF, python-docx, and mammoth"""
    
    def __init__(self):
        try:
            import fitz  # PyMuPDF
            from docx import Document
            import mammoth
            
            self.fitz = fitz
            self.Document = Document
            self.mammoth = mammoth
            logger.info("Document converter initialized successfully")
        except ImportError as e:
            logger.error(f"Required libraries not installed: {str(e)}")
            raise AppException("Document conversion libraries not available", 500, str(e))
        except Exception as e:
            logger.error(f"Failed to initialize document converter: {str(e)}")
            raise AppException("Failed to initialize document converter", 500, str(e))

    async def convert_to_markdown(self, file_path: Path) -> str:
        """Convert document to markdown format"""
        try:
            file_ext = file_path.suffix.lower()
            
            # Validate file type
            if file_ext not in ['.pdf', '.docx', '.doc', '.txt']:
                raise AppException(f"Unsupported file type: {file_ext}", 400)
            
            # Run conversion in thread pool to avoid blocking
            def _convert_sync():
                try:
                    if file_ext == '.pdf':
                        return self._convert_pdf(file_path)
                    elif file_ext == '.docx':
                        return self._convert_docx(file_path)
                    elif file_ext == '.doc':
                        return self._convert_doc(file_path)
                    elif file_ext == '.txt':
                        return self._convert_txt(file_path)
                except Exception as e:
                    logger.error(f"Document conversion failed for {file_path}: {str(e)}")
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

    def _convert_pdf(self, file_path: Path) -> str:
        """Convert PDF to markdown using PyMuPDF"""
        doc = self.fitz.open(str(file_path))
        text = ""
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():
                text += f"\n## Page {page_num + 1}\n\n{page_text}\n"
        
        doc.close()
        
        if not text.strip():
            raise AppException("No text content found in PDF", 400)
            
        return f"# {file_path.stem}\n\n{text}"

    def _convert_docx(self, file_path: Path) -> str:
        """Convert DOCX to markdown using mammoth"""
        with open(file_path, "rb") as docx_file:
            result = self.mammoth.convert_to_markdown(docx_file)
            
            if result.messages:
                for message in result.messages:
                    logger.warning(f"Mammoth conversion warning: {message}")
            
            markdown_content = result.value
            
            if not markdown_content.strip():
                # Fallback to python-docx if mammoth fails
                return self._convert_docx_fallback(file_path)
                
            return markdown_content

    def _convert_docx_fallback(self, file_path: Path) -> str:
        """Fallback DOCX conversion using python-docx"""
        doc = self.Document(str(file_path))
        text = ""
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n\n"
        
        if not text.strip():
            raise AppException("No text content found in DOCX", 400)
            
        return f"# {file_path.stem}\n\n{text}"

    def _convert_doc(self, file_path: Path) -> str:
        """Convert DOC files - requires python-docx2txt or similar"""
        try:
            import docx2txt
            text = docx2txt.process(str(file_path))
            
            if not text.strip():
                raise AppException("No text content found in DOC file", 400)
                
            return f"# {file_path.stem}\n\n{text}"
        except ImportError:
            # Fallback: suggest user to convert DOC to DOCX
            raise AppException("DOC files not supported. Please convert to DOCX format.", 400)

    def _convert_txt(self, file_path: Path) -> str:
        """Convert TXT to markdown"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        if not content.strip():
            raise AppException("Text file is empty", 400)
            
        return f"# {file_path.stem}\n\n{content}"