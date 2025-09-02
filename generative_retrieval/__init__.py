"""
Generative Retrieval Data Library

A simple abstraction layer built on top of Huggingface datasets for generative retrieval tasks.
"""

from . import data, models
from .data import GRData, DocID
from .models import DSIModel, DocIDTokenizer, TrieConstraint

__version__ = "0.1.0"
__all__ = ['data', 'models', 'GRData', 'DocID', 'DSIModel', 'DocIDTokenizer', 'TrieConstraint']