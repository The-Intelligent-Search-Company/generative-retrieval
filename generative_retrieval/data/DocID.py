"""
Enhanced DocID Assignment System

This module provides improved DocID assignment methods that handle duplicate documents
elegantly while maintaining consistency for DSI training.

Key improvements:
1. Content-based DocID assignment (identical content → identical DocID) - NOW DEFAULT
2. Deduplication with query aggregation
3. Backward compatibility with existing sequential method
4. Configurable DocID formats and collision handling
"""

from typing import Any, Dict, Callable, Union, List
import hashlib
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def generate_sequential_ids(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Original sequential DocID assignment (for backward compatibility)."""
    example['doc_id'] = f"{idx:08d}"
    return example


def generate_content_based_ids(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Generate DocIDs based on document content hash.
    
    This ensures identical documents get identical DocIDs, solving the
    duplicate content problem in DSI training.
    
    Args:
        example: Dataset example containing 'doc_text'
        idx: Index (unused, kept for compatibility)
        
    Returns:
        Example with content-based doc_id added
    """
    doc_text = example.get('doc_text', '')
    
    # Create stable hash of document content
    content_hash = hashlib.md5(doc_text.encode('utf-8')).hexdigest()
    
    # Convert to 8-digit numeric format for consistency
    # Take first 8 hex chars and convert to decimal, then format
    hex_segment = content_hash[:8]
    numeric_id = int(hex_segment, 16) % 100000000  # Ensure 8 digits max
    doc_id = f"{numeric_id:08d}"
    
    example['doc_id'] = doc_id
    return example


def generate_normalized_content_ids(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Generate DocIDs based on normalized document content.
    
    This version normalizes text before hashing to handle minor variations
    in whitespace, punctuation, etc.
    """
    doc_text = example.get('doc_text', '')
    
    # Normalize text (lowercase, strip whitespace, normalize punctuation)
    normalized = _normalize_text_for_hashing(doc_text)
    
    # Create hash of normalized content
    content_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
    hex_segment = content_hash[:8]
    numeric_id = int(hex_segment, 16) % 100000000
    doc_id = f"{numeric_id:08d}"
    
    example['doc_id'] = doc_id
    return example


def generate_sha256_content_ids(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Generate DocIDs using SHA256 for better collision resistance."""
    doc_text = example.get('doc_text', '')
    
    # Use SHA256 for better collision resistance
    content_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()
    hex_segment = content_hash[:8]
    numeric_id = int(hex_segment, 16) % 100000000
    doc_id = f"{numeric_id:08d}"
    
    example['doc_id'] = doc_id
    return example


def _normalize_text_for_hashing(text: str) -> str:
    """Normalize text for consistent hashing."""
    import re
    
    # Convert to lowercase
    text = text.lower()
    
    # Normalize whitespace (multiple spaces/tabs/newlines → single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def deduplicate_dataset_with_query_aggregation(dataset) -> Dict[str, Any]:
    """
    Advanced deduplication that aggregates queries for duplicate documents.
    
    This approach:
    1. Groups identical documents together
    2. Combines all queries that reference each document
    3. Creates single DocID per unique document
    4. Preserves all query-document relationships
    """
    logger.info("Starting dataset deduplication with query aggregation...")
    
    # Convert dataset to dictionary for processing
    if hasattr(dataset, 'to_dict'):
        data_dict = dataset.to_dict()
    else:
        data_dict = dataset
    
    doc_texts = data_dict.get('doc_text', [])
    queries = data_dict.get('query', [])
    
    if len(doc_texts) != len(queries):
        raise ValueError("doc_text and query lists must have same length")
    
    # Group queries by document content
    doc_to_queries = defaultdict(set)  # Use set to avoid duplicate queries
    doc_to_metadata = {}  # Store additional metadata per document
    
    for i, (doc_text, query) in enumerate(zip(doc_texts, queries)):
        doc_to_queries[doc_text].add(query)
        
        # Store first occurrence metadata (if any additional fields exist)
        if doc_text not in doc_to_metadata:
            doc_to_metadata[doc_text] = {
                key: value[i] for key, value in data_dict.items() 
                if key not in ['doc_text', 'query', 'doc_id']
            }
    
    # Create deduplicated dataset
    unique_docs = []
    unique_queries = []
    doc_query_counts = []
    
    total_original = len(doc_texts)
    total_unique = len(doc_to_queries)
    
    for doc_text, query_set in doc_to_queries.items():
        # For each unique document, create entries for all its queries
        for query in query_set:
            unique_docs.append(doc_text)
            unique_queries.append(query)
            doc_query_counts.append(len(query_set))
    
    # Create result dictionary
    result = {
        'doc_text': unique_docs,
        'query': unique_queries,
        'doc_query_count': doc_query_counts,  # How many queries reference this document
    }
    
    # Add back any additional metadata fields
    for field_name in data_dict:
        if field_name not in ['doc_text', 'query', 'doc_id']:
            result[field_name] = []
            for doc_text in unique_docs:
                result[field_name].append(doc_to_metadata[doc_text].get(field_name, None))
    
    logger.info(f"Deduplication complete:")
    logger.info(f"  Original entries: {total_original}")
    logger.info(f"  Unique documents: {total_unique}")
    logger.info(f"  Final entries: {len(unique_docs)} (after query expansion)")
    logger.info(f"  Reduction: {total_original - len(unique_docs)} entries")
    
    return result


def analyze_duplicate_content(dataset) -> Dict[str, Any]:
    """
    Analyze the dataset for duplicate content and provide statistics.
    
    Args:
        dataset: HuggingFace dataset or dictionary
        
    Returns:
        Analysis results including duplicate statistics
    """
    if hasattr(dataset, 'to_dict'):
        data_dict = dataset.to_dict()
    else:
        data_dict = dataset
    
    doc_texts = data_dict.get('doc_text', [])
    
    # Count document frequencies
    from collections import Counter
    doc_counts = Counter(doc_texts)
    
    total_entries = len(doc_texts)
    unique_docs = len(doc_counts)
    duplicate_docs = sum(1 for count in doc_counts.values() if count > 1)
    duplicate_entries = sum(count - 1 for count in doc_counts.values() if count > 1)
    
    # Find most duplicated documents
    most_duplicated = doc_counts.most_common(5)
    
    analysis = {
        'total_entries': total_entries,
        'unique_documents': unique_docs,
        'duplicate_documents': duplicate_docs,
        'duplicate_entries': duplicate_entries,
        'duplication_rate': duplicate_entries / total_entries if total_entries > 0 else 0,
        'most_duplicated': [
            {'doc_text': doc[:100] + '...' if len(doc) > 100 else doc, 'count': count}
            for doc, count in most_duplicated
        ],
        'potential_docid_conflicts_sequential': duplicate_entries,
        'efficiency_gain_content_based': duplicate_entries,
    }
    
    return analysis


def add_doc_ids(dataset, method: Union[str, Callable] = "content_based"):
    """
    Enhanced DocID assignment with multiple methods.
    
    Args:
        dataset: HuggingFace dataset
        method: DocID assignment method
            - "content_based": Hash-based assignment using MD5 (DEFAULT - recommended)
            - "sequential": Original index-based assignment (backward compatible)
            - "content_normalized": Hash-based with text normalization
            - "content_sha256": Hash-based using SHA256
            - "deduplicated": Deduplicate first, then assign sequential IDs
            - Callable: Custom function
    
    Returns:
        Dataset with doc_id column added
    """
    if isinstance(method, str):
        if method == "content_based":
            logger.info("Using content-based DocID assignment (MD5) - recommended for DSI")
            return dataset.map(generate_content_based_ids, with_indices=True)
        
        elif method == "sequential":
            logger.info("Using sequential DocID assignment - may cause conflicts with duplicate documents")
            return dataset.map(generate_sequential_ids, with_indices=True)
        
        elif method == "content_normalized":
            logger.info("Using normalized content-based DocID assignment")
            return dataset.map(generate_normalized_content_ids, with_indices=True)
        
        elif method == "content_sha256":
            logger.info("Using SHA256 content-based DocID assignment")  
            return dataset.map(generate_sha256_content_ids, with_indices=True)
        
        elif method == "deduplicated":
            logger.info("Using deduplication with sequential DocID assignment")
            # First deduplicate the data
            from datasets import Dataset
            dedup_data = deduplicate_dataset_with_query_aggregation(dataset)
            dedup_dataset = Dataset.from_dict(dedup_data)
            # Then assign sequential IDs to deduplicated data
            return dedup_dataset.map(generate_sequential_ids, with_indices=True)
        
        else:
            available_methods = [
                "content_based", "sequential", "content_normalized", 
                "content_sha256", "deduplicated"
            ]
            raise ValueError(
                f"Unsupported doc_id generation method: '{method}'. "
                f"Available methods: {available_methods}"
            )
    
    elif callable(method):
        # Custom function provided - use it directly
        return dataset.map(method, with_indices=True)
    
    else:
        raise TypeError(f"Method must be a string or callable function, got {type(method)}")