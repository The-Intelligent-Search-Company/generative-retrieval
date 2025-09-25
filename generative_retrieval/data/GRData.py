"""
Enhanced GRData class with improved DocID assignment capabilities.

This version addresses the duplicate document issue by providing
multiple DocID assignment strategies and built-in analysis.
"""

from typing import Union, Optional, Dict, Any, List, Callable
from datasets import Dataset, load_dataset
import pandas as pd
import logging
from .DocID import add_doc_ids, analyze_duplicate_content

logger = logging.getLogger(__name__)


class GRData:
    """
    Enhanced GRData class with improved DocID assignment capabilities.
    
    Key improvements over previous version:
    1. Content-based DocID assignment by default (solves duplicate document issue)
    2. Built-in duplicate analysis and handling  
    3. Enhanced training readiness validation
    4. Backward compatibility maintained
    
    Supports two schemas:
    - Schema 1: doc_id, doc_text, query
    - Schema 2: doc_text, query (doc_id will be auto-generated)
    
    Compatible with all Huggingface dataset functions and file formats.
    """
    
    def __init__(self, dataset: Optional[Dataset] = None):
        self.dataset = dataset
        self._duplicate_analysis = None
    
    @classmethod
    def from_file(cls, 
                  file_path: str, 
                  format: Optional[str] = None, 
                  doc_id_method: Union[str, Callable] = "content_based",
                  analyze_duplicates: bool = True,
                  **kwargs) -> 'GRData':
        """
        Create GRData from file with enhanced DocID assignment.
        
        Args:
            file_path: Path to data file
            format: File format (json, csv, parquet, or auto-detect)
            doc_id_method: DocID assignment method
                - "content_based": Hash-based using MD5 (DEFAULT - recommended)
                - "sequential": Original index-based (backward compatible)
                - "content_normalized": Hash-based with text normalization
                - "content_sha256": Hash-based using SHA256
                - "deduplicated": Deduplicate first, then sequential IDs
            analyze_duplicates: Whether to analyze duplicates before assignment
            **kwargs: Additional arguments for load_dataset
        """
        if format is None:
            if file_path.endswith('.json'):
                format = 'json'
            elif file_path.endswith('.csv'):
                format = 'csv'
            elif file_path.endswith('.parquet'):
                format = 'parquet'
                
        if format:
            dataset = load_dataset(format, data_files=file_path, **kwargs)['train']
        else:
            dataset = load_dataset(file_path, **kwargs)['train']
            
        return cls.from_dataset(
            dataset, 
            doc_id_method=doc_id_method,
            analyze_duplicates=analyze_duplicates
        )
    
    @classmethod
    def from_dataset(cls, 
                    dataset: Dataset, 
                    doc_id_method: Union[str, Callable] = "content_based",
                    analyze_duplicates: bool = True) -> 'GRData':
        """
        Create GRData from HuggingFace dataset.
        
        Args:
            dataset: HuggingFace dataset
            doc_id_method: DocID assignment method (content_based is default)
            analyze_duplicates: Whether to analyze duplicates
        """
        gr_data = cls(dataset)
        
        # Analyze duplicates before DocID assignment if requested
        if analyze_duplicates and 'doc_id' not in dataset.column_names:
            gr_data._duplicate_analysis = analyze_duplicate_content(dataset)
            gr_data._log_duplicate_analysis()
        
        # Add DocIDs if not present
        if 'doc_id' not in dataset.column_names:
            gr_data.dataset = add_doc_ids(gr_data.dataset, method=doc_id_method)
            logger.info(f"DocIDs assigned using method: {doc_id_method}")
        else:
            logger.info("DocIDs already present in dataset")
            
        return gr_data
    
    @classmethod
    def from_dict(cls, 
                 data: Dict[str, List], 
                 doc_id_method: Union[str, Callable] = "content_based",
                 analyze_duplicates: bool = True) -> 'GRData':
        """Create GRData from dictionary with content-based DocIDs by default."""
        dataset = Dataset.from_dict(data)
        return cls.from_dataset(
            dataset, 
            doc_id_method=doc_id_method, 
            analyze_duplicates=analyze_duplicates
        )
    
    @classmethod
    def from_pandas(cls, 
                   df: pd.DataFrame, 
                   doc_id_method: Union[str, Callable] = "content_based",
                   analyze_duplicates: bool = True) -> 'GRData':
        """Create GRData from pandas DataFrame with content-based DocIDs by default.""" 
        dataset = Dataset.from_pandas(df)
        return cls.from_dataset(
            dataset, 
            doc_id_method=doc_id_method,
            analyze_duplicates=analyze_duplicates
        )
    
    
    def _log_duplicate_analysis(self):
        """Log duplicate content analysis results."""
        if self._duplicate_analysis is None:
            return
            
        analysis = self._duplicate_analysis
        logger.info("=== DUPLICATE CONTENT ANALYSIS ===")
        logger.info(f"Total entries: {analysis['total_entries']}")
        logger.info(f"Unique documents: {analysis['unique_documents']}")
        logger.info(f"Documents with duplicates: {analysis['duplicate_documents']}")
        logger.info(f"Duplicate entries: {analysis['duplicate_entries']}")
        logger.info(f"Duplication rate: {analysis['duplication_rate']:.2%}")
        
        if analysis['duplicate_entries'] > 0:
            logger.warning(f"Found {analysis['duplicate_entries']} duplicate entries!")
            logger.warning("This would cause DocID conflicts with sequential assignment")
            logger.info("Using content-based DocID assignment to resolve conflicts")
            
            if analysis['most_duplicated']:
                logger.info("Most duplicated documents:")
                for item in analysis['most_duplicated'][:3]:
                    logger.info(f"  '{item['doc_text'][:60]}...' (appears {item['count']} times)")
        else:
            logger.info("No duplicate documents found - all content is unique")
    
    def get_duplicate_analysis(self) -> Optional[Dict[str, Any]]:
        """Get the duplicate content analysis results."""
        return self._duplicate_analysis
    
    def analyze_docid_consistency(self) -> Dict[str, Any]:
        """
        Analyze DocID assignment consistency.
        
        Returns statistics about DocID assignment, including whether
        identical documents have identical DocIDs.
        """
        if self.dataset is None:
            return {'error': 'No dataset loaded'}
        
        data_dict = self.dataset.to_dict()
        doc_texts = data_dict.get('doc_text', [])
        doc_ids = data_dict.get('doc_id', [])
        
        if not doc_texts or not doc_ids:
            return {'error': 'Missing doc_text or doc_id columns'}
        
        # Group DocIDs by document content
        from collections import defaultdict
        content_to_docids = defaultdict(set)
        docid_to_contents = defaultdict(set)
        
        for doc_text, doc_id in zip(doc_texts, doc_ids):
            content_to_docids[doc_text].add(doc_id)
            docid_to_contents[doc_id].add(doc_text)
        
        # Analyze consistency
        consistent_mappings = sum(1 for docids in content_to_docids.values() if len(docids) == 1)
        inconsistent_mappings = sum(1 for docids in content_to_docids.values() if len(docids) > 1)
        
        unique_docids = len(docid_to_contents)
        docid_collisions = sum(1 for contents in docid_to_contents.values() if len(contents) > 1)
        
        return {
            'total_documents': len(content_to_docids),
            'total_docids': unique_docids,
            'consistent_mappings': consistent_mappings,
            'inconsistent_mappings': inconsistent_mappings,
            'docid_collisions': docid_collisions,
            'is_consistent': inconsistent_mappings == 0 and docid_collisions == 0,
            'consistency_rate': consistent_mappings / len(content_to_docids) if content_to_docids else 1.0
        }
    
    def print_training_summary(self):
        """Print a comprehensive training readiness summary."""
        if self.dataset is None:
            print("❌ No dataset loaded")
            return
        
        print("=" * 60)
        print("DSI TRAINING DATASET SUMMARY")
        print("=" * 60)
        
        schema_info = self.get_schema_info()
        docid_consistency = self.analyze_docid_consistency()
        
        data_dict = self.dataset.to_dict()
        doc_texts = data_dict.get('doc_text', [])
        queries = data_dict.get('query', [])
        
        print(f"Dataset Size: {len(doc_texts)} examples")
        print(f"Schema: {schema_info['schema_type']}")
        print(f"Valid Schema: {'✅' if schema_info['is_valid'] else '❌'}")
        
        # DocID consistency
        print(f"\nDocID Assignment:")
        print(f"  Unique documents: {docid_consistency['total_documents']}")
        print(f"  Unique DocIDs: {docid_consistency['total_docids']}")
        print(f"  Consistent mappings: {docid_consistency['consistent_mappings']}")
        print(f"  Inconsistent mappings: {docid_consistency['inconsistent_mappings']}")
        print(f"  DocID consistency: {'✅' if docid_consistency['is_consistent'] else '❌'}")
        
        # Duplicate analysis
        if self._duplicate_analysis:
            dup = self._duplicate_analysis
            print(f"\nDuplicate Content:")
            print(f"  Duplicate entries: {dup['duplicate_entries']}")
            print(f"  Duplication rate: {dup['duplication_rate']:.2%}")
            if dup['duplicate_entries'] > 0:
                print("  Status: ⚠️  Duplicates found (handled by content-based DocIDs)")
            else:
                print("  Status: ✅ No duplicates")
        
        # Text statistics
        if doc_texts and queries:
            import numpy as np
            doc_lengths = [len(text.split()) for text in doc_texts]
            query_lengths = [len(query.split()) for query in queries]
            print(f"\nText Statistics:")
            print(f"  Document length: {np.mean(doc_lengths):.1f} ± {np.std(doc_lengths):.1f} words")
            print(f"  Query length: {np.mean(query_lengths):.1f} ± {np.std(query_lengths):.1f} words")
        
        # Training readiness
        print(f"\nTraining Readiness:")
        print(f"  Required columns: {'✅' if schema_info['is_valid'] else '❌'}")
        print(f"  DocID consistency: {'✅' if docid_consistency['is_consistent'] else '❌'}")
        
        ready_for_training = (
            schema_info['is_valid'] and 
            docid_consistency['is_consistent'] and
            len(doc_texts) > 0
        )
        print(f"  Ready for training: {'✅' if ready_for_training else '❌'}")
        
        print("=" * 60)
    
    def validate_schema(self) -> bool:
        required_columns = {'doc_id', 'doc_text', 'query'}
        actual_columns = set(self.dataset.column_names)
        return required_columns.issubset(actual_columns)
    
    def get_schema_info(self) -> Dict[str, Any]:
        columns = self.dataset.column_names
        num_rows = len(self.dataset)
        schema_type = None
        if set(columns) >= {'doc_id', 'doc_text', 'query'}:
            schema_type = "Schema 1 (doc_id, doc_text, query)"
        elif set(columns) >= {'doc_text', 'query'}:
            schema_type = "Schema 2 (doc_text, query) - doc_id needed"
        return {
            'columns': columns,
            'num_rows': num_rows,
            'schema_type': schema_type,
            'is_valid': self.validate_schema()
        }
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if self.dataset is None:
            raise AttributeError(f"No dataset loaded")
        return getattr(self.dataset, name)
    
    def __len__(self):
        return len(self.dataset) if self.dataset else 0
    
    def __getitem__(self, key):
        if self.dataset is None:
            raise IndexError("No dataset loaded")
        return self.dataset[key]
    
    def __iter__(self):
        if self.dataset is None:
            raise StopIteration
        return iter(self.dataset)
    
    def push_to_hub(self, repo_id: str, **kwargs):
        return self.dataset.push_to_hub(repo_id, **kwargs)
    
    def save_to_disk(self, dataset_path: str, **kwargs):
        return self.dataset.save_to_disk(dataset_path, **kwargs)
    
    def to_pandas(self) -> pd.DataFrame:
        return self.dataset.to_pandas()
    
    def to_dict(self) -> Dict:
        return self.dataset.to_dict()
    
    def filter(self, function, **kwargs):
        filtered_dataset = self.dataset.filter(function, **kwargs)
        new_instance = GRData(filtered_dataset)
        new_instance._duplicate_analysis = self._duplicate_analysis  # Preserve analysis
        return new_instance
    
    def map(self, function, **kwargs):
        mapped_dataset = self.dataset.map(function, **kwargs)
        new_instance = GRData(mapped_dataset)
        new_instance._duplicate_analysis = self._duplicate_analysis  # Preserve analysis
        return new_instance
    
    def select(self, indices):
        selected_dataset = self.dataset.select(indices)
        new_instance = GRData(selected_dataset)
        new_instance._duplicate_analysis = self._duplicate_analysis  # Preserve analysis  
        return new_instance
    
    def train_test_split(self, test_size=0.2, **kwargs):
        split_dataset = self.dataset.train_test_split(test_size=test_size, **kwargs)
        train_instance = GRData(split_dataset['train'])
        test_instance = GRData(split_dataset['test'])
        
        # Preserve analysis in both splits
        train_instance._duplicate_analysis = self._duplicate_analysis
        test_instance._duplicate_analysis = self._duplicate_analysis
        
        return {
            'train': train_instance,
            'test': test_instance
        }
