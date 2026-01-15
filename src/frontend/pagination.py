"""
Memory-efficient pagination utilities for 1GB RAM constraint
"""
from typing import List, Any, Generator
import math

class MemoryEfficientPaginator:
    """
    Paginator that loads data in chunks to save memory
    """
    
    def __init__(self, data_source, page_size: int = 10, max_memory_mb: int = 50):
        self.data_source = data_source
        self.page_size = page_size
        self.max_memory_mb = max_memory_mb
        self.current_page = 0
        self.total_items = 0
        self.total_pages = 0
        
        # Estimate memory usage per item (bytes)
        self.estimated_item_size = 1024  # 1KB default
    
    def get_page(self, page_number: int) -> List[Any]:
        """Get a specific page of data"""
        if hasattr(self.data_source, '__getitem__'):
            # For list-like sources
            start = (page_number - 1) * self.page_size
            end = start + self.page_size
            return self.data_source[start:end]
        elif callable(self.data_source):
            # For callable sources (like API calls)
            return self.data_source(page_number, self.page_size)
        else:
            raise ValueError("Unsupported data source type")
    
    def get_page_generator(self) -> Generator[List[Any], None, None]:
        """Generate pages one at a time to save memory"""
        page = 1
        while True:
            data = self.get_page(page)
            if not data:
                break
            yield data
            page += 1
    
    def estimate_memory_usage(self, data: List[Any]) -> int:
        """Estimate memory usage of data in bytes"""
        # Simple estimation: items * avg size
        return len(data) * self.estimated_item_size
    
    def is_within_memory_limit(self, data: List[Any]) -> bool:
        """Check if data is within memory limit"""
        estimated_bytes = self.estimate_memory_usage(data)
        estimated_mb = estimated_bytes / (1024 * 1024)
        return estimated_mb <= self.max_memory_mb
    
    def get_optimal_page_size(self) -> int:
        """Calculate optimal page size based on memory limit"""
        max_items = (self.max_memory_mb * 1024 * 1024) // self.estimated_item_size
        return min(self.page_size, max_items)

class LazyDataLoader:
    """
    Lazy loader that loads data on-demand
    """
    
    def __init__(self, load_function, cache_size: int = 5):
        self.load_function = load_function
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size
    
    def get(self, key, *args, **kwargs):
        """Get data, loading if not in cache"""
        if key in self.cache:
            # Move to front (most recently used)
            self.cache_order.remove(key)
            self.cache_order.insert(0, key)
            return self.cache[key]
        
        # Load data
        data = self.load_function(*args, **kwargs)
        
        # Cache management
        if len(self.cache) >= self.cache_size:
            # Remove least recently used
            lru_key = self.cache_order.pop()
            del self.cache[lru_key]
        
        # Add to cache
        self.cache[key] = data
        self.cache_order.insert(0, key)
        
        return data
    
    def clear_cache(self):
        """Clear cache to free memory"""
        self.cache.clear()
        self.cache_order.clear()

# Example usage for Streamlit
def create_streamlit_pagination(data, key_prefix="page"):
    """Create Streamlit-compatible pagination"""
    import streamlit as st
    
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 1
    
    paginator = MemoryEfficientPaginator(data)
    
    # Display pagination controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    
    with col1:
        if st.button("⏮️ First", key=f"{key_prefix}_first"):
            st.session_state.page_number = 1
            st.rerun()
    
    with col2:
        if st.button("◀️ Prev", key=f"{key_prefix}_prev") and st.session_state.page_number > 1:
            st.session_state.page_number -= 1
            st.rerun()
    
    with col3:
        if st.button("Next ▶️", key=f"{key_prefix}_next"):
            st.session_state.page_number += 1
            st.rerun()
    
    with col4:
        if st.button("Last ⏭️", key=f"{key_prefix}_last"):
            st.session_state.page_number = paginator.total_pages
            st.rerun()
    
    # Display current page info
    st.caption(f"Page {st.session_state.page_number} of {paginator.total_pages}")
    
    # Get and display current page data
    return paginator.get_page(st.session_state.page_number)
