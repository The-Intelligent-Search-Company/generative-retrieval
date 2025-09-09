"""
Generative Retrieval Data Library

A modular, production-ready library for generative retrieval tasks with DSI models.
"""

from . import data, models, train
from .data import GRData, DocID
from .models import DSIModel, DocIDTokenizer, TrieConstraint
from .train import DSITrainer, TrainingConfig, ConfigPresets, TrainingUtils, MetricsTracker

__version__ = "0.1.0"
__all__ = [
    'data', 'models', 'train',
    'GRData', 'DocID', 
    'DSIModel', 'DocIDTokenizer', 'TrieConstraint',
    'DSITrainer', 'TrainingConfig', 'ConfigPresets', 'TrainingUtils', 'MetricsTracker'
]