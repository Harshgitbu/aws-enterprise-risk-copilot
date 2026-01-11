"""
Document processor with memory-efficient chunking
"""
import hashlib
from typing import List, Dict, Any
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Memory-efficient document chunk"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    document_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'metadata': self.metadata,
            'chunk_id': self.chunk_id,
            'document_id': self.document_id
        }

class DocumentProcessor:
    """
    Processes documents with memory constraints
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Split document into chunks with memory efficiency
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of document chunks
        """
        if not text:
            return []
        
        # Generate document ID from metadata or text hash
        document_id = metadata.get('document_id', 
                                 hashlib.md5(text.encode()).hexdigest()[:16])
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate chunk end
            end = start + self.chunk_size
            
            # Try to end at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for boundary in ['. ', '! ', '? ', '\n\n', '\n']:
                    boundary_pos = text.rfind(boundary, start, end)
                    if boundary_pos != -1:
                        end = boundary_pos + len(boundary)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Skip empty chunks
                chunk_id = f"{document_id}_chunk{chunk_index}"
                
                chunk = DocumentChunk(
                    text=chunk_text,
                    metadata={
                        **metadata,
                        'chunk_index': chunk_index,
                        'start_pos': start,
                        'end_pos': end
                    },
                    chunk_id=chunk_id,
                    document_id=document_id
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.overlap if end - self.overlap > start else end
        
        logger.info(f"Chunked document into {len(chunks)} chunks")
        return chunks
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Process multiple documents
        
        Args:
            documents: List of dicts with 'text' and 'metadata'
            
        Returns:
            List of all document chunks
        """
        all_chunks = []
        
        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            if text:
                chunks = self.chunk_document(text, metadata)
                all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
    
    def estimate_memory_usage(self, num_chunks: int, avg_chunk_size: int = 500) -> int:
        """
        Estimate memory usage for chunks
        
        Args:
            num_chunks: Number of chunks
            avg_chunk_size: Average chunk size in characters
            
        Returns:
            Estimated memory in bytes
        """
        # Rough estimate: each character ~1 byte + metadata overhead
        text_memory = num_chunks * avg_chunk_size
        metadata_memory = num_chunks * 500  # Approximate metadata overhead
        return text_memory + metadata_memory

if __name__ == "__main__":
    # Test document processor
    processor = DocumentProcessor(chunk_size=300, overlap=30)
    
    test_documents = [
        {
            'text': '''
            Enterprise risk management is a critical function for modern businesses.
            It involves identifying, assessing, and prioritizing risks followed by
            coordinated application of resources to minimize or control probability
            and impact of unfortunate events. AI technologies can enhance risk
            intelligence by analyzing large datasets and identifying patterns that
            humans might miss.
            
            Risk intelligence platforms should provide real-time monitoring,
            predictive analytics, and automated reporting capabilities. These systems
            help organizations make informed decisions and maintain compliance with
            regulatory requirements.
            ''',
            'metadata': {
                'source': 'test_source',
                'title': 'Enterprise Risk Management Overview',
                'type': 'article'
            }
        }
    ]
    
    print("Testing document processor...")
    chunks = processor.process_documents(test_documents)
    
    print(f"\nGenerated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Text preview: {chunk.text[:100]}...")
        print(f"  Metadata: {chunk.metadata}")
    
    # Estimate memory
    estimated_memory = processor.estimate_memory_usage(len(chunks))
    print(f"\nEstimated memory for chunks: {estimated_memory/1024:.1f} KB")
    print("✅ Document processor test complete!")
