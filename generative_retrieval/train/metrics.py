from typing import Dict, List, Any, Optional
import json
import os
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class MetricsTracker:
    def __init__(self):
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
        self.epoch_metrics: Dict[int, Dict[str, float]] = {}
        self.step_metrics: Dict[int, Dict[str, float]] = {}
        self.start_time = time.time()
        self.best_metrics: Dict[str, float] = {}
    
    def log_step_metrics(self, step: int, metrics: Dict[str, float]):
        self.step_metrics[step] = metrics.copy()
        
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
            
            if key not in self.best_metrics:
                self.best_metrics[key] = value
            else:
                if "loss" in key.lower():
                    self.best_metrics[key] = min(self.best_metrics[key], value)
                else:
                    self.best_metrics[key] = max(self.best_metrics[key], value)
    
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        self.epoch_metrics[epoch] = metrics.copy()
        
        for key, value in metrics.items():
            if key not in self.best_metrics:
                self.best_metrics[key] = value
            else:
                if "loss" in key.lower():
                    self.best_metrics[key] = min(self.best_metrics[key], value)
                else:
                    self.best_metrics[key] = max(self.best_metrics[key], value)
    
    def get_current_metrics(self) -> Dict[str, float]:
        if not self.epoch_metrics:
            return {}
        
        latest_epoch = max(self.epoch_metrics.keys())
        return self.epoch_metrics[latest_epoch]
    
    def get_best_metrics(self) -> Dict[str, float]:
        return self.best_metrics.copy()
    
    def get_metrics_history(self, metric_name: str) -> List[float]:
        return self.metrics_history[metric_name].copy()
    
    def get_all_metrics(self) -> Dict[str, Any]:
        return {
            'metrics_history': dict(self.metrics_history),
            'epoch_metrics': self.epoch_metrics,
            'step_metrics': self.step_metrics,
            'best_metrics': self.best_metrics,
            'start_time': self.start_time
        }
    
    def load_metrics(self, metrics_data: Dict[str, Any]):
        self.metrics_history = defaultdict(list, metrics_data.get('metrics_history', {}))
        self.epoch_metrics = metrics_data.get('epoch_metrics', {})
        self.step_metrics = metrics_data.get('step_metrics', {})
        self.best_metrics = metrics_data.get('best_metrics', {})
        self.start_time = metrics_data.get('start_time', time.time())
    
    def save_metrics(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        metrics_data = self.get_all_metrics()
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
    
    def load_metrics_from_file(self, filepath: str):
        with open(filepath, 'r') as f:
            metrics_data = json.load(f)
        self.load_metrics(metrics_data)
    
    def get_training_time(self) -> float:
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        summary = {
            'total_epochs': len(self.epoch_metrics),
            'total_steps': len(self.step_metrics),
            'training_time_seconds': self.get_training_time(),
            'best_metrics': self.get_best_metrics(),
            'current_metrics': self.get_current_metrics()
        }
        
        if self.metrics_history:
            summary['metrics_available'] = list(self.metrics_history.keys())
        
        return summary
    
    def print_summary(self):
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("TRAINING METRICS SUMMARY")
        print("=" * 60)
        
        print(f"Total Epochs: {summary['total_epochs']}")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Training Time: {summary['training_time_seconds']:.2f} seconds")
        
        if summary['best_metrics']:
            print("\nBest Metrics:")
            for key, value in summary['best_metrics'].items():
                print(f"  {key}: {value:.4f}")
        
        if summary['current_metrics']:
            print("\nCurrent Metrics:")
            for key, value in summary['current_metrics'].items():
                print(f"  {key}: {value:.4f}")
        
        print("=" * 60)
    
    def plot_metrics(self, metrics_names: Optional[List[str]] = None, save_path: Optional[str] = None):
        if not self.epoch_metrics:
            print("No epoch metrics available for plotting")
            return
        
        if metrics_names is None:
            metrics_names = list(set().union(*[m.keys() for m in self.epoch_metrics.values()]))
        
        available_metrics = metrics_names
        if not available_metrics:
            print("No metrics available for plotting")
            return
        
        n_metrics = len(available_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_metrics == 1 else axes
        else:
            axes = axes.flatten()
        
        epochs = sorted(self.epoch_metrics.keys())
        
        for i, metric_name in enumerate(available_metrics):
            if i >= len(axes):
                break
                
            values = []
            for epoch in epochs:
                if metric_name in self.epoch_metrics[epoch]:
                    values.append(self.epoch_metrics[epoch][metric_name])
                else:
                    values.append(None)
            
            valid_epochs = [e for e, v in zip(epochs, values) if v is not None]
            valid_values = [v for v in values if v is not None]
            
            if valid_values:
                axes[i].plot(valid_epochs, valid_values, marker='o', linewidth=2, markersize=4)
                axes[i].set_title(f'{metric_name}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric_name)
                axes[i].grid(True, alpha=0.3)
                
                if "loss" in metric_name.lower():
                    axes[i].set_ylabel('Loss')
                elif any(word in metric_name.lower() for word in ['accuracy', 'hits', 'precision', 'recall', 'f1']):
                    axes[i].set_ylabel('Score')
        
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved to: {save_path}")
        
        plt.show()
    
    def get_learning_curve_data(self) -> Dict[str, List[float]]:
        learning_curve = {}
        
        if self.epoch_metrics:
            epochs = sorted(self.epoch_metrics.keys())
            
            for metric_name in ['train_total_loss', 'eval_total_loss', 'train_loss', 'eval_loss']:
                values = []
                for epoch in epochs:
                    if metric_name in self.epoch_metrics[epoch]:
                        values.append(self.epoch_metrics[epoch][metric_name])
                    else:
                        values.append(None)
                
                if any(v is not None for v in values):
                    learning_curve[metric_name] = values
        
        return learning_curve
    
    def detect_overfitting(self, patience: int = 3, min_delta: float = 0.001) -> Dict[str, Any]:
        if len(self.epoch_metrics) < patience + 1:
            return {'overfitting_detected': False, 'message': 'Insufficient data'}
        
        epochs = sorted(self.epoch_metrics.keys())
        
        train_losses = []
        eval_losses = []
        
        for epoch in epochs:
            metrics = self.epoch_metrics[epoch]
            train_loss = metrics.get('train_total_loss') or metrics.get('train_loss')
            eval_loss = metrics.get('eval_total_loss') or metrics.get('eval_loss')
            
            if train_loss is not None:
                train_losses.append(train_loss)
            if eval_loss is not None:
                eval_losses.append(eval_loss)
        
        if len(train_losses) < patience + 1 or len(eval_losses) < patience + 1:
            return {'overfitting_detected': False, 'message': 'Insufficient loss data'}
        
        recent_train = train_losses[-patience:]
        recent_eval = eval_losses[-patience:]
        
        train_improving = all(
            recent_train[i] - recent_train[i+1] > min_delta 
            for i in range(len(recent_train)-1)
        )
        
        eval_degrading = all(
            recent_eval[i+1] - recent_eval[i] > min_delta 
            for i in range(len(recent_eval)-1)
        )
        
        overfitting_detected = train_improving and eval_degrading
        
        return {
            'overfitting_detected': overfitting_detected,
            'train_improving': train_improving,
            'eval_degrading': eval_degrading,
            'patience': patience,
            'min_delta': min_delta,
            'recent_train_losses': recent_train,
            'recent_eval_losses': recent_eval
        }
    
    def export_to_csv(self, filepath: str):
        import pandas as pd
        
        if not self.epoch_metrics:
            print("No epoch metrics to export")
            return
        
        epochs = sorted(self.epoch_metrics.keys())
        data_rows = []
        
        for epoch in epochs:
            row = {'epoch': epoch}
            row.update(self.epoch_metrics[epoch])
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Metrics exported to: {filepath}")
    
    def compare_with_baseline(self, baseline_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        current_metrics = self.get_best_metrics()
        comparison = {}
        
        for metric_name in baseline_metrics:
            if metric_name in current_metrics:
                baseline_val = baseline_metrics[metric_name]
                current_val = current_metrics[metric_name]
                
                if "loss" in metric_name.lower():
                    improvement = baseline_val - current_val
                    improvement_pct = (improvement / baseline_val) * 100 if baseline_val != 0 else 0
                else:
                    improvement = current_val - baseline_val
                    improvement_pct = (improvement / baseline_val) * 100 if baseline_val != 0 else 0
                
                comparison[metric_name] = {
                    'baseline': baseline_val,
                    'current': current_val,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                }
        
        return comparison
    
    def get_epoch_summary(self, epoch: int) -> Optional[Dict[str, Any]]:
        if epoch not in self.epoch_metrics:
            return None
        
        metrics = self.epoch_metrics[epoch]
        summary = {
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary