"""
Generative Retrieval Data Library

A simple abstraction layer built on top of Huggingface datasets for generative retrieval tasks.
"""

from . import data
from .data import GRData, DocID

__version__ = "0.1.0"
__all__ = ['data', 'GRData', 'DocID']