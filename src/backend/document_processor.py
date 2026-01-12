"""
Memory-efficient document processing for RAG pipeline
OPTIMIZED FOR: 1GB RAM, streaming processing, chunking strategies
"""

import os
import re
import hashlib
from typing import List, Dict, Any, Generator, Optional
from dataclasses import dataclass
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Memory-efficient document chunk"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    token_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
            "token_count": self.token_count
        }


class MemoryEfficientDocumentProcessor:
    """
    Processes documents with minimal memory footprint
    Key features:
    - Streaming processing (never load entire doc in memory)
    - Adaptive chunking based on content
    - Overlap for context preservation
    - Metadata preservation
    """
    
    def __init__(self, max_chunk_size: int = 500, overlap: int = 50):
        """
        Initialize processor with memory constraints
        
        Args:
            max_chunk_size: Maximum characters per chunk (for 1GB RAM)
            overlap: Overlap between chunks for context
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        
        # Pre-compiled regex for efficiency
        self.sentence_splitter = re.compile(r'(?<=[.!?])\s+')
        self.paragraph_splitter = re.compile(r'\n\s*\n')
        
        logger.info(f"📄 DocumentProcessor initialized: {max_chunk_size} chars/chunk")
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def create_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Create unique chunk ID"""
        return f"{doc_id}_chunk_{chunk_index}"
    
    def smart_chunking(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Smart chunking that preserves semantic boundaries
        Memory-efficient: processes in streaming fashion
        """
        chunks = []
        
        # If text is small enough, return as single chunk
        if len(text) <= self.max_chunk_size:
            chunk_id = self.create_chunk_id(metadata.get("doc_id", "unknown"), 0)
            chunks.append(DocumentChunk(
                text=text,
                metadata=metadata.copy(),
                chunk_id=chunk_id,
                token_count=self.estimate_tokens(text)
            ))
            return chunks
        
        # Try to split by paragraphs first
        paragraphs = self.paragraph_splitter.split(text)
        
        current_chunk = ""
        current_metadata = metadata.copy()
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph itself is too large, split by sentences
            if len(para) > self.max_chunk_size:
                sentences = self.sentence_splitter.split(para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if len(current_chunk) + len(sentence) + 1 <= self.max_chunk_size:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        # Save current chunk
                        if current_chunk:
                            chunk_id = self.create_chunk_id(
                                metadata.get("doc_id", "unknown"), 
                                chunk_index
                            )
                            chunks.append(DocumentChunk(
                                text=current_chunk,
                                metadata=current_metadata.copy(),
                                chunk_id=chunk_id,
                                token_count=self.estimate_tokens(current_chunk)
                            ))
                            chunk_index += 1
                        
                        # Start new chunk with overlap
                        current_chunk = sentence
            else:
                # Check if adding paragraph would exceed chunk size
                if len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                    current_chunk += "\n\n" + para if current_chunk else para
                else:
                    # Save current chunk
                    if current_chunk:
                        chunk_id = self.create_chunk_id(
                            metadata.get("doc_id", "unknown"), 
                            chunk_index
                        )
                        chunks.append(DocumentChunk(
                            text=current_chunk,
                            metadata=current_metadata.copy(),
                            chunk_id=chunk_id,
                            token_count=self.estimate_tokens(current_chunk)
                        ))
                        chunk_index += 1
                    
                    # Start new chunk
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_id = self.create_chunk_id(
                metadata.get("doc_id", "unknown"), 
                chunk_index
            )
            chunks.append(DocumentChunk(
                text=current_chunk,
                metadata=current_metadata.copy(),
                chunk_id=chunk_id,
                token_count=self.estimate_tokens(current_chunk)
            ))
        
        # Add overlap between chunks for context
        if len(chunks) > 1 and self.overlap > 0:
            chunks = self._add_overlap(chunks)
        
        logger.info(f"📊 Created {len(chunks)} chunks from document")
        return chunks
    
    def _add_overlap(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Add overlap between chunks for better context"""
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk gets overlap from next
                if i + 1 < len(chunks):
                    next_text = chunks[i + 1].text
                    overlap_text = next_text[:self.overlap]
                    new_text = chunk.text + "\n[CONTINUED...] " + overlap_text
                else:
                    new_text = chunk.text
            elif i == len(chunks) - 1:
                # Last chunk gets overlap from previous
                prev_text = chunks[i - 1].text
                overlap_text = prev_text[-self.overlap:]
                new_text = "[...CONTINUATION] " + overlap_text + "\n" + chunk.text
            else:
                # Middle chunks get overlap from both sides
                prev_text = chunks[i - 1].text
                next_text = chunks[i + 1].text
                prev_overlap = prev_text[-self.overlap//2:]
                next_overlap = next_text[:self.overlap//2]
                new_text = "[...CONTINUATION] " + prev_overlap + "\n" + \
                          chunk.text + "\n[CONTINUED...] " + next_overlap
            
            overlapped_chunks.append(DocumentChunk(
                text=new_text,
                metadata=chunk.metadata.copy(),
                chunk_id=chunk.chunk_id,
                token_count=self.estimate_tokens(new_text)
            ))
        
        return overlapped_chunks
    
    def process_file_streaming(self, file_path: str, 
                             metadata: Optional[Dict] = None) -> Generator[DocumentChunk, None, None]:
        """
        Process file in streaming mode (never load entire file)
        Generator yields chunks one by one
        """
        if metadata is None:
            metadata = {}
        
        # Add file info to metadata
        metadata.update({
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "doc_id": hashlib.md5(file_path.encode()).hexdigest()[:16]
        })
        
        try:
            # Read file in chunks for memory efficiency
            with open(file_path, 'r', encoding='utf-8') as f:
                buffer = ""
                chunk_index = 0
                
                for line in f:
                    buffer += line
                    
                    # When buffer reaches chunk size, process it
                    if len(buffer) >= self.max_chunk_size * 10:  # Process in batches
                        chunks = self.smart_chunking(buffer, metadata)
                        for chunk in chunks:
                            chunk.chunk_id = self.create_chunk_id(
                                metadata["doc_id"], 
                                chunk_index
                            )
                            chunk_index += 1
                            yield chunk
                        buffer = ""  # Reset buffer
                
                # Process remaining buffer
                if buffer.strip():
                    chunks = self.smart_chunking(buffer, metadata)
                    for chunk in chunks:
                        chunk.chunk_id = self.create_chunk_id(
                            metadata["doc_id"], 
                            chunk_index
                        )
                        chunk_index += 1
                        yield chunk
            
            logger.info(f"✅ Processed {file_path} in streaming mode")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            yield DocumentChunk(
                text=f"Error processing document: {str(e)}",
                metadata=metadata,
                chunk_id="error_0",
                token_count=0
            )
    
    def process_text(self, text: str, metadata: Optional[Dict] = None) -> List[DocumentChunk]:
        """Process text string (for API inputs)"""
        if metadata is None:
            metadata = {}
        
        metadata["doc_id"] = hashlib.md5(text.encode()).hexdigest()[:16]
        metadata["source"] = "api_input"
        
        return self.smart_chunking(text, metadata)


# Default processor instance
default_processor = MemoryEfficientDocumentProcessor(
    max_chunk_size=500,  # Conservative for 1GB RAM
    overlap=50
)
