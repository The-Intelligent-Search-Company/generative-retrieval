import os
import random
import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import DataLoader
import json
import shutil
from pathlib import Path
import time


class TrainingUtils:
    
    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    @staticmethod
    def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
        logger = logging.getLogger("DSITraining")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        if not logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            if log_file:
                os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'cached_gb': torch.cuda.memory_reserved() / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
                'max_cached_gb': torch.cuda.max_memory_reserved() / 1e9
            }
        else:
            return {'message': 'CUDA not available'}
    
    @staticmethod
    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def estimate_training_time(
        num_examples: int,
        batch_size: int,
        num_epochs: int,
        seconds_per_batch: float
    ) -> Dict[str, float]:
        batches_per_epoch = num_examples / batch_size
        total_batches = batches_per_epoch * num_epochs
        total_seconds = total_batches * seconds_per_batch
        
        return {
            'batches_per_epoch': batches_per_epoch,
            'total_batches': total_batches,
            'estimated_seconds': total_seconds,
            'estimated_minutes': total_seconds / 60,
            'estimated_hours': total_seconds / 3600
        }
    
    @staticmethod
    def create_output_directory(base_path: str, experiment_name: Optional[str] = None) -> str:
        if experiment_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        output_dir = os.path.join(base_path, experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        
        subdirs = ['checkpoints', 'logs', 'metrics', 'plots', 'config']
        for subdir in subdirs:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        
        return output_dir
    
    @staticmethod
    def save_training_info(
        output_dir: str,
        config: Dict[str, Any],
        model_info: Dict[str, Any],
        data_info: Dict[str, Any]
    ):
        info = {
            'config': config,
            'model_info': model_info,
            'data_info': data_info,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
        
        info_path = os.path.join(output_dir, 'config', 'training_info.json')
        os.makedirs(os.path.dirname(info_path), exist_ok=True)
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)
    
    @staticmethod
    def cleanup_checkpoints(
        checkpoint_dir: str,
        keep_best: bool = True,
        keep_last_n: int = 3,
        keep_every_n_epochs: int = 10
    ):
        if not os.path.exists(checkpoint_dir):
            return
        
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint-')]
        if not checkpoint_files:
            return
        
        files_to_keep = set()
        
        if keep_best and 'best_model.pt' in os.listdir(checkpoint_dir):
            files_to_keep.add('best_model.pt')
        
        checkpoint_files.sort(key=lambda x: int(x.split('-')[2].split('.')[0]))
        
        if keep_last_n > 0:
            files_to_keep.update(checkpoint_files[-keep_last_n:])
        
        if keep_every_n_epochs > 0:
            for i, filename in enumerate(checkpoint_files):
                step = int(filename.split('-')[2].split('.')[0])
                if step % (keep_every_n_epochs * 1000) == 0:
                    files_to_keep.add(filename)
        
        for filename in checkpoint_files:
            if filename not in files_to_keep:
                file_path = os.path.join(checkpoint_dir, filename)
                os.remove(file_path)
    
    @staticmethod
    def validate_data_consistency(train_data, eval_data=None) -> Dict[str, Any]:
        results = {
            'train_valid': True,
            'eval_valid': True,
            'consistency_valid': True,
            'issues': []
        }
        
        train_dict = train_data.to_dict()
        required_columns = ['doc_text', 'query', 'doc_id']
        
        for col in required_columns:
            if col not in train_dict:
                results['train_valid'] = False
                results['issues'].append(f"Missing column '{col}' in training data")
        
        if results['train_valid']:
            lengths = [len(train_dict[col]) for col in required_columns]
            if not all(l == lengths[0] for l in lengths):
                results['train_valid'] = False
                results['issues'].append("Inconsistent column lengths in training data")
        
        if eval_data is not None:
            eval_dict = eval_data.to_dict()
            
            for col in required_columns:
                if col not in eval_dict:
                    results['eval_valid'] = False
                    results['issues'].append(f"Missing column '{col}' in evaluation data")
            
            if results['eval_valid']:
                lengths = [len(eval_dict[col]) for col in required_columns]
                if not all(l == lengths[0] for l in lengths):
                    results['eval_valid'] = False
                    results['issues'].append("Inconsistent column lengths in evaluation data")
            
            if results['train_valid'] and results['eval_valid']:
                train_docids = set(train_dict['doc_id'])
                eval_docids = set(eval_dict['doc_id'])
                
                overlap = train_docids.intersection(eval_docids)
                if overlap:
                    results['consistency_valid'] = False
                    results['issues'].append(f"Found {len(overlap)} overlapping doc_ids between train and eval")
        
        results['overall_valid'] = all([
            results['train_valid'],
            results['eval_valid'],
            results['consistency_valid']
        ])
        
        return results
    
    @staticmethod
    def compute_data_statistics(data) -> Dict[str, Any]:
        data_dict = data.to_dict()
        
        doc_text_lengths = [len(text.split()) for text in data_dict['doc_text']]
        query_lengths = [len(query.split()) for query in data_dict['query']]
        docid_lengths = [len(docid) for docid in data_dict['doc_id']]
        
        return {
            'num_examples': len(data_dict['doc_text']),
            'doc_text_stats': {
                'mean_length': np.mean(doc_text_lengths),
                'std_length': np.std(doc_text_lengths),
                'min_length': np.min(doc_text_lengths),
                'max_length': np.max(doc_text_lengths),
                'median_length': np.median(doc_text_lengths)
            },
            'query_stats': {
                'mean_length': np.mean(query_lengths),
                'std_length': np.std(query_lengths),
                'min_length': np.min(query_lengths),
                'max_length': np.max(query_lengths),
                'median_length': np.median(query_lengths)
            },
            'docid_stats': {
                'mean_length': np.mean(docid_lengths),
                'std_length': np.std(docid_lengths),
                'min_length': np.min(docid_lengths),
                'max_length': np.max(docid_lengths),
                'unique_docids': len(set(data_dict['doc_id']))
            }
        }
    
    @staticmethod
    def benchmark_dataloader(
        dataloader: DataLoader,
        num_batches: int = 10,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, float]:
        times = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            start_time = time.time()
            
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        value.to(device)
            elif isinstance(batch, torch.Tensor):
                batch.to(device)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'avg_batch_time': np.mean(times),
            'std_batch_time': np.std(times),
            'min_batch_time': np.min(times),
            'max_batch_time': np.max(times),
            'total_time': np.sum(times)
        }
    
    @staticmethod
    def create_backup(source_dir: str, backup_dir: str):
        if os.path.exists(source_dir):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"backup_{timestamp}")
            shutil.copytree(source_dir, backup_path)
            return backup_path
        return None
    
    @staticmethod
    def monitor_gpu_utilization():
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            if gpus:
                gpu = gpus[0]
                return {
                    'gpu_utilization': gpu.load * 100,
                    'memory_utilization': gpu.memoryUtil * 100,
                    'memory_used_gb': gpu.memoryUsed / 1024,
                    'memory_total_gb': gpu.memoryTotal / 1024,
                    'temperature': gpu.temperature
                }
        except ImportError:
            pass
        
        return {'message': 'GPUtil not available'}
    
    @staticmethod
    def format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    @staticmethod
    def format_bytes(bytes_value: float) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f}{unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f}PB"
    
    @staticmethod
    def check_dependencies() -> Dict[str, Any]:
        dependencies = {
            'torch': torch.__version__,
            'transformers': None,
            'datasets': None,
            'numpy': np.__version__,
            'matplotlib': None,
            'pandas': None,
            'GPUtil': None
        }
        
        try:
            import transformers
            dependencies['transformers'] = transformers.__version__
        except ImportError:
            dependencies['transformers'] = 'Not installed'
        
        try:
            import datasets
            dependencies['datasets'] = datasets.__version__
        except ImportError:
            dependencies['datasets'] = 'Not installed'
        
        try:
            import matplotlib
            dependencies['matplotlib'] = matplotlib.__version__
        except ImportError:
            dependencies['matplotlib'] = 'Not installed'
        
        try:
            import pandas
            dependencies['pandas'] = pandas.__version__
        except ImportError:
            dependencies['pandas'] = 'Not installed'
        
        try:
            import GPUtil
            dependencies['GPUtil'] = 'Available'
        except ImportError:
            dependencies['GPUtil'] = 'Not installed'
        
        return dependencies
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        import platform
        
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        
        return info