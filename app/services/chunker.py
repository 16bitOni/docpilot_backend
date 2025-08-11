import re
import os
import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pinecone import Pinecone
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

class ChunkType(Enum):
    SECTION = "section"
    SUBSECTION = "subsection" 
    CONTENT = "content"
    LIST = "list"
    TABLE = "table"

@dataclass
class ChunkMetadata:
    chunk_id: str
    chunk_type: ChunkType
    filename: str  # Added filename field
    start_line: int
    end_line: int
    section_path: List[str]
    section_level: int

@dataclass
class DocumentChunk:
    content: str
    metadata: ChunkMetadata

class MarkdownChunkerStorage:
    def __init__(self, index_name: str = None, max_chunk_size: int = 800):
        self.max_chunk_size = max_chunk_size
        self.index = self._get_vector_store(index_name)
        
        # Patterns for markdown elements
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.list_pattern = re.compile(r'^(\s*[-*+]|\s*\d+\.)\s+(.+)$', re.MULTILINE)
    
    def process_and_store_file(self, file_path: str, namespace: str = "") -> Dict[str, Any]:
        """
        Process markdown file, chunk it, and store in Pinecone
        
        Args:
            file_path: Path to the markdown file
            namespace: Pinecone namespace to store in
            
        Returns:
            Dictionary with processing results
        """
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        filename = os.path.basename(file_path)
        
        # Chunk the markdown
        chunks = self._chunk_document(content, filename)
        
        # Store in Pinecone
        self._store_chunks(chunks, namespace)
        
        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "chunk_types": self._get_chunk_type_stats(chunks),
            "status": "success"
        }
    
    def process_and_store_content(self, content: str, namespace: str = "", filename: str = "document") -> Dict[str, Any]:
        """
        Process markdown content directly, chunk it, and store in Pinecone
        
        Args:
            content: Markdown content as string
            namespace: Pinecone namespace to store in
            filename: Name to use for the document
            
        Returns:
            Dictionary with processing results
        """
        # Chunk the markdown content
        chunks = self._chunk_document(content, filename)
        
        # Store in Pinecone
        self._store_chunks(chunks, namespace)
        
        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "chunk_types": self._get_chunk_type_stats(chunks),
            "status": "success"
        }
    
    def _get_vector_store(self, index_name: str = None) -> Pinecone:
        """Initialize Pinecone index"""
        # Load config
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = config["vector_store"]["environment"]
        
        if not pinecone_api_key:
            raise ValueError("Missing Pinecone API key")
        if not pinecone_env:
            raise ValueError("Missing Pinecone environment in config")
        
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = index_name or config["vector_store"]["index_name"]
        
        # Create index if it doesn't exist
        if not pc.has_index(index_name):
            pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region=pinecone_env,
                embed={
                    "model": "multilingual-e5-large",
                    "field_map": {"text": "chunk_text"}
                }
            )
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
        
        return pc.Index(index_name)
    
    def _chunk_document(self, markdown_content: str, filename: str) -> List[DocumentChunk]:
        """Chunk markdown document"""
        lines = markdown_content.split('\n')
        chunks = []
        current_section_path = []
        chunk_counter = 0
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Update section path
                current_section_path = self._update_section_path(current_section_path, title, level)
                
                # Create header chunk
                chunk = DocumentChunk(
                    content=line,
                    metadata=ChunkMetadata(
                        chunk_id=f"chunk_{chunk_counter}",
                        chunk_type=ChunkType.SECTION if level <= 2 else ChunkType.SUBSECTION,
                        filename=filename,
                        start_line=i + 1,
                        end_line=i + 1,
                        section_path=current_section_path.copy(),
                        section_level=level
                    )
                )
                chunks.append(chunk)
                chunk_counter += 1
                i += 1
                continue
            
            # Check for tables
            if '|' in line and i < len(lines) - 1:
                table_chunk, lines_consumed = self._extract_table_chunk(
                    lines, i, current_section_path, chunk_counter, filename
                )
                if table_chunk:
                    chunks.append(table_chunk)
                    chunk_counter += 1
                    i += lines_consumed
                    continue
            
            # Check for lists
            if re.match(r'^(\s*[-*+]|\s*\d+\.)\s+', line):
                list_chunk, lines_consumed = self._extract_list_chunk(
                    lines, i, current_section_path, chunk_counter, filename
                )
                if list_chunk:
                    chunks.append(list_chunk)
                    chunk_counter += 1
                    i += lines_consumed
                    continue
            
            # Regular content
            if line:
                content_chunk, lines_consumed = self._extract_content_chunk(
                    lines, i, current_section_path, chunk_counter, filename
                )
                if content_chunk:
                    chunks.append(content_chunk)
                    chunk_counter += 1
                    i += lines_consumed
                    continue
            
            i += 1
        
        return chunks
    
    def _update_section_path(self, current_path: List[str], title: str, level: int) -> List[str]:
        """Update section hierarchy path"""
        if level == 1:
            return [title]
        elif level == 2:
            return [title]
        else:
            if len(current_path) >= 1:
                return [current_path[0], title]
            else:
                return [title]
    
    def _extract_table_chunk(self, lines: List[str], start_idx: int, 
                           section_path: List[str], chunk_id: int, filename: str) -> Tuple[Optional[DocumentChunk], int]:
        """Extract table as a single chunk"""
        table_lines = []
        i = start_idx
        
        while i < len(lines) and ('|' in lines[i] or lines[i].strip() == ''):
            if '|' in lines[i]:
                table_lines.append(lines[i])
            elif lines[i].strip() == '' and table_lines:
                break
            i += 1
        
        if not table_lines:
            return None, 1
        
        content = '\n'.join(table_lines)
        
        chunk = DocumentChunk(
            content=content,
            metadata=ChunkMetadata(
                chunk_id=f"chunk_{chunk_id}",
                chunk_type=ChunkType.TABLE,
                filename=filename,
                start_line=start_idx + 1,
                end_line=start_idx + len(table_lines),
                section_path=section_path.copy(),
                section_level=0
            )
        )
        
        return chunk, len(table_lines)
    
    def _extract_list_chunk(self, lines: List[str], start_idx: int,
                          section_path: List[str], chunk_id: int, filename: str) -> Tuple[Optional[DocumentChunk], int]:
        """Extract list as a single chunk"""
        list_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
                
            if re.match(r'^(\s*)([-*+]|\d+\.)\s+(.+)$', line):
                list_lines.append(lines[i])
                i += 1
            else:
                break
        
        if not list_lines:
            return None, 1
        
        content = '\n'.join(list_lines)
        
        chunk = DocumentChunk(
            content=content,
            metadata=ChunkMetadata(
                chunk_id=f"chunk_{chunk_id}",
                chunk_type=ChunkType.LIST,
                filename=filename,
                start_line=start_idx + 1,
                end_line=start_idx + len(list_lines),
                section_path=section_path.copy(),
                section_level=0
            )
        )
        
        return chunk, len(list_lines)
    
    def _extract_content_chunk(self, lines: List[str], start_idx: int,
                             section_path: List[str], chunk_id: int, filename: str) -> Tuple[Optional[DocumentChunk], int]:
        """Extract regular content chunk"""
        content_lines = []
        current_size = 0
        i = start_idx
        
        while i < len(lines) and current_size < self.max_chunk_size:
            line = lines[i]
            
            # Stop at headers, lists, or tables
            if (re.match(r'^#{1,6}\s+', line.strip()) or 
                re.match(r'^(\s*[-*+]|\s*\d+\.)\s+', line.strip()) or 
                '|' in line):
                break
            
            if line.strip():
                content_lines.append(line)
                current_size += len(line)
            elif content_lines:
                content_lines.append(line)
            
            i += 1
        
        if not content_lines:
            return None, 1
        
        # Clean up trailing empty lines
        while content_lines and not content_lines[-1].strip():
            content_lines.pop()
        
        if not content_lines:
            return None, i - start_idx
        
        content = '\n'.join(content_lines)
        
        chunk = DocumentChunk(
            content=content,
            metadata=ChunkMetadata(
                chunk_id=f"chunk_{chunk_id}",
                chunk_type=ChunkType.CONTENT,
                filename=filename,
                start_line=start_idx + 1,
                end_line=start_idx + len(content_lines),
                section_path=section_path.copy(),
                section_level=0
            )
        )
        
        return chunk, len(content_lines)
    
    def _store_chunks(self, chunks: List[DocumentChunk], namespace: str):
        """Store chunks in Pinecone"""
        records = []
        
        for chunk in chunks:
            record = {
                "_id": f"{chunk.metadata.filename}_{chunk.metadata.chunk_id}",
                "chunk_text": chunk.content,
                "filename": chunk.metadata.filename,
                "chunk_id": chunk.metadata.chunk_id,
                "chunk_type": chunk.metadata.chunk_type.value,
                "start_line": chunk.metadata.start_line,
                "end_line": chunk.metadata.end_line,
                "section_path": " > ".join(chunk.metadata.section_path),
                "section_level": chunk.metadata.section_level,
                "content_length": len(chunk.content),
                "timestamp": int(time.time())
            }
            records.append(record)
        
        # Upsert in batches
        batch_size = 90
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.index.upsert_records(namespace, batch)
            time.sleep(1)
    
    def _get_chunk_type_stats(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """Get statistics about chunk types"""
        stats = {}
        for chunk in chunks:
            chunk_type = chunk.metadata.chunk_type.value
            stats[chunk_type] = stats.get(chunk_type, 0) + 1
        return stats


# Usage example
if __name__ == "__main__":
    # Initialize the chunker
    chunker = MarkdownChunkerStorage("hr-policy-index")
    
    # Process and store a markdown file
    result = chunker.process_and_store_file(
        file_path="converted_markdown\hr2.md",
        namespace="documents"
    )
    
    print("Processing Results:")
    print(f"Filename: {result['filename']}")
    print(f"Total Chunks: {result['total_chunks']}")
    print(f"Chunk Types: {result['chunk_types']}")
    print(f"Status: {result['status']}")