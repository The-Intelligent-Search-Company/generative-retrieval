from typing import Union, Optional, Dict, Any, List, Callable
from datasets import Dataset, load_dataset
import pandas as pd
from .DocID import add_doc_ids

class GRData:
    """
    A simple abstraction layer built on top of Huggingface datasets for generative retrieval.
    
    Supports two schemas:
    - Schema 1: doc_id, doc_text, query
    - Schema 2: doc_text, query (doc_id will be auto-generated)
    
    Compatible with all Huggingface dataset functions and file formats.
    """
    
    def __init__(self, dataset: Optional[Dataset] = None):
        self.dataset = dataset
    
    @classmethod
    def from_file(cls, 
                  file_path: str, 
                  format: Optional[str] = None, 
                  doc_id_method: Union[str, Callable] = "sequential",
                  **kwargs) -> 'GRData':
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
        gr_data = cls(dataset)
        if 'doc_id' not in dataset.column_names:
            gr_data.dataset = add_doc_ids(gr_data.dataset, method=doc_id_method)
        return gr_data
    
    @classmethod
    def from_dataset(cls, dataset: Dataset, doc_id_method: Union[str, Callable] = "sequential") -> 'GRData':
        gr_data = cls(dataset)
        if 'doc_id' not in dataset.column_names:
            gr_data.dataset = add_doc_ids(gr_data.dataset, method=doc_id_method)
        return gr_data
    
    @classmethod
    def from_dict(cls, data: Dict[str, List], doc_id_method: Union[str, Callable] = "sequential") -> 'GRData':
        dataset = Dataset.from_dict(data)
        return cls.from_dataset(dataset, doc_id_method=doc_id_method)
    
    @classmethod
    def from_pandas(cls, df: pd.DataFrame, doc_id_method: Union[str, Callable] = "sequential") -> 'GRData':
        dataset = Dataset.from_pandas(df)
        return cls.from_dataset(dataset, doc_id_method=doc_id_method)
    
    
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
        return GRData(filtered_dataset)
    
    def map(self, function, **kwargs):
        mapped_dataset = self.dataset.map(function, **kwargs)
        return GRData(mapped_dataset)
    
    def select(self, indices):
        selected_dataset = self.dataset.select(indices)
        return GRData(selected_dataset)
    
    def train_test_split(self, test_size=0.2, **kwargs):
        split_dataset = self.dataset.train_test_split(test_size=test_size, **kwargs)
        return {
            'train': GRData(split_dataset['train']),
            'test': GRData(split_dataset['test'])
        }
