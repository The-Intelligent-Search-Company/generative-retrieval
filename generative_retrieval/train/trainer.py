from typing import List, Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
import os
import json
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup
import random

from ..models import DSIModel
from ..data import GRData
from .config import TrainingConfig
from .metrics import MetricsTracker
from .utils import TrainingUtils


class DSIDataset(Dataset):
    def __init__(
        self, 
        gr_data: GRData, 
        task_type: str = "both",
        max_input_length: int = 512,
        max_target_length: int = 20,
        indexing_ratio: float = 1.0
    ):
        self.gr_data = gr_data
        self.task_type = task_type
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.indexing_ratio = indexing_ratio
        
        self.data_dict = gr_data.to_dict()
        self.doc_texts = self.data_dict['doc_text']
        self.queries = self.data_dict['query']
        self.doc_ids = self.data_dict['doc_id']
        
        self.examples = self._prepare_examples()
    
    def _prepare_examples(self) -> List[Dict[str, Any]]:
        examples = []
        
        for i, (doc_text, query, doc_id) in enumerate(zip(self.doc_texts, self.queries, self.doc_ids)):
            if self.task_type in ["indexing", "both"]:
                # Add multiple indexing examples based on ratio
                num_indexing = max(1, int(self.indexing_ratio))
                for _ in range(num_indexing):
                    examples.append({
                        'input_text': f"index document: {doc_text}",
                        'target_text': doc_id,
                        'task_type': 'indexing',
                        'original_idx': i
                    })
            
            if self.task_type in ["retrieval", "both"]:
                examples.append({
                    'input_text': f"retrieve query: {query}",
                    'target_text': doc_id,
                    'task_type': 'retrieval',
                    'original_idx': i
                })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class DSITrainer:
    """Trainer class for DSI (Differentiable Search Index) models.
    
    Handles both single-phase and two-phase training strategies for DSI models,
    including proper multi-task loss weighting and evaluation.
    
    Args:
        model: DSIModel instance to train
        config: Training configuration
        train_data: Training dataset in GRData format
        eval_data: Optional evaluation dataset
        logger: Optional custom logger
    """
    
    def __init__(
        self,
        model: DSIModel,
        config: TrainingConfig,
        train_data: GRData,
        eval_data: Optional[GRData] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.eval_data = eval_data
        self.logger = logger or self._setup_logger()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
        self.model.to(self.device)
        
        self.optimizer = self._setup_optimizer()
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        self.metrics_tracker = MetricsTracker()
        self.global_step = 0
        self.current_epoch = 0
        
        self._setup_dataloaders()
        self._setup_scheduler()
        
        self.logger.info(f"DSITrainer initialized with {self.model.get_model_size():,} parameters")
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Training examples: {len(self.train_dataset)}")
        if self.eval_data:
            self.logger.info(f"Evaluation examples: {len(self.eval_dataset)}")
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("DSITrainer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_optimizer(self) -> optim.Optimizer:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        if self.config.optimizer == "adamw":
            return optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-6
            )
        elif self.config.optimizer == "adam":
            return optim.Adam(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate
            )
        elif self.config.optimizer == "adafactor":
            try:
                from transformers.optimization import Adafactor
                return Adafactor(
                    optimizer_grouped_parameters,
                    lr=self.config.learning_rate,
                    scale_parameter=False,
                    relative_step=False
                )
            except ImportError:
                self.logger.warning("Adafactor not available, falling back to AdamW")
                return optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=self.config.learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-6
                )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def _setup_dataloaders(self):
        self.train_dataset = DSIDataset(
            self.train_data,
            task_type=self.config.task_type,
            max_input_length=self.config.max_input_length,
            max_target_length=self.config.max_target_length,
            indexing_ratio=getattr(self.config, 'indexing_ratio', 1.0)
        )
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        if self.eval_data:
            self.eval_dataset = DSIDataset(
                self.eval_data,
                task_type=self.config.task_type,
                max_input_length=self.config.max_input_length,
                max_target_length=self.config.max_target_length,
                indexing_ratio=1.0  # Always 1:1 for evaluation
            )
            
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=False,
                collate_fn=self._collate_fn,
                num_workers=self.config.num_workers,
                pin_memory=True if self.device.type == "cuda" else False
            )
    
    def _setup_scheduler(self):
        if self.config.scheduler == "linear":
            total_steps = len(self.train_dataloader) * self.config.num_epochs
            warmup_steps = int(total_steps * self.config.warmup_ratio)
            
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif self.config.scheduler == "cosine":
            total_steps = len(self.train_dataloader) * self.config.num_epochs
            warmup_steps = int(total_steps * self.config.warmup_ratio)
            
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=total_steps - warmup_steps,
                eta_min=self.config.learning_rate * 0.1
            )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_texts = [item['input_text'] for item in batch]
        target_texts = [item['target_text'] for item in batch]
        
        input_encodings = self.model.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_input_length,
            return_tensors="pt"
        )
        
        target_encodings = self.model.tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_target_length,
            return_tensors="pt"
        )
        
        labels = target_encodings['input_ids'].clone()
        labels[labels == self.model.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': labels,
            'task_types': [item['task_type'] for item in batch]
        }
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_losses = {'total': 0.0, 'indexing': 0.0, 'retrieval': 0.0}
        epoch_counts = {'total': 0, 'indexing': 0, 'retrieval': 0}
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
            leave=False
        )
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Separate batch by task type for proper multi-task training
            task_types = batch['task_types']
            indexing_mask = [t == 'indexing' for t in task_types]
            retrieval_mask = [t == 'retrieval' for t in task_types]
            
            indexing_batch = None
            retrieval_batch = None
            
            if any(indexing_mask):
                indexing_indices = [i for i, mask in enumerate(indexing_mask) if mask]
                indexing_batch = {
                    'input_ids': batch['input_ids'][indexing_indices],
                    'attention_mask': batch['attention_mask'][indexing_indices],
                    'labels': batch['labels'][indexing_indices]
                }
            
            if any(retrieval_mask):
                retrieval_indices = [i for i, mask in enumerate(retrieval_mask) if mask]
                retrieval_batch = {
                    'input_ids': batch['input_ids'][retrieval_indices],
                    'attention_mask': batch['attention_mask'][retrieval_indices],
                    'labels': batch['labels'][retrieval_indices]
                }
            
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    if hasattr(self.model, 'compute_multitask_loss') and (indexing_batch is not None and retrieval_batch is not None):
                        outputs = self.model.compute_multitask_loss(
                            indexing_batch=indexing_batch,
                            retrieval_batch=retrieval_batch,
                            indexing_weight=self.config.indexing_weight,
                            retrieval_weight=self.config.retrieval_weight
                        )
                        loss = outputs.loss
                    else:
                        # Fallback to regular training
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels']
                        )
                        loss = outputs.loss
            else:
                if hasattr(self.model, 'compute_multitask_loss') and (indexing_batch is not None and retrieval_batch is not None):
                    outputs = self.model.compute_multitask_loss(
                        indexing_batch=indexing_batch,
                        retrieval_batch=retrieval_batch,
                        indexing_weight=self.config.indexing_weight,
                        retrieval_weight=self.config.retrieval_weight
                    )
                    loss = outputs.loss
                else:
                    # Fallback to regular training
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss
            
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
            
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            epoch_losses['total'] += loss.item()
            epoch_counts['total'] += 1
            
            # Properly track task-specific losses
            if hasattr(outputs, 'loss_components'):
                for task, task_loss in outputs.loss_components.items():
                    epoch_losses[task] += task_loss.item()
                    epoch_counts[task] += 1
            else:
                # Fallback: distribute loss across task types
                for task_type in set(batch['task_types']):
                    task_count = sum(1 for t in batch['task_types'] if t == task_type)
                    epoch_losses[task_type] += loss.item() * (task_count / len(batch['task_types']))
                    epoch_counts[task_type] += task_count
            
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.max_grad_norm > 0:
                    if self.config.use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.max_grad_norm
                        )
                        self.optimizer.step()
                else:
                    if self.config.use_mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            if self.config.logging_steps > 0 and self.global_step % self.config.logging_steps == 0:
                self._log_training_stats(loss.item(), current_lr)
            
            if (self.config.eval_steps > 0 and 
                self.global_step % self.config.eval_steps == 0 and 
                self.eval_data is not None):
                try:
                    eval_metrics = self.evaluate()
                    self.model.train()
                except Exception as e:
                    self.logger.warning(f"Evaluation failed during training: {e}")
            
            if (self.config.save_steps > 0 and 
                self.global_step % self.config.save_steps == 0):
                self._save_checkpoint()
        
        avg_losses = {}
        for key in epoch_losses:
            if epoch_counts[key] > 0:
                avg_losses[f'train_{key}_loss'] = epoch_losses[key] / epoch_counts[key]
            else:
                avg_losses[f'train_{key}_loss'] = 0.0
        
        return avg_losses
    
    def evaluate(self) -> Dict[str, float]:
        if self.eval_data is None:
            self.logger.warning("No evaluation data provided")
            return {}
        
        self.model.eval()
        eval_losses = {'total': 0.0, 'indexing': 0.0, 'retrieval': 0.0}
        eval_counts = {'total': 0, 'indexing': 0, 'retrieval': 0}
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss.item()
                eval_losses['total'] += loss
                eval_counts['total'] += 1
                
                for task_type in batch['task_types']:
                    eval_losses[task_type] += loss / len(batch['task_types'])
                    eval_counts[task_type] += 1
        
        avg_losses = {}
        for key in eval_losses:
            if eval_counts[key] > 0:
                avg_losses[f'eval_{key}_loss'] = eval_losses[key] / eval_counts[key]
            else:
                avg_losses[f'eval_{key}_loss'] = 0.0
        
        if self.config.compute_retrieval_metrics:
            retrieval_metrics = self._compute_retrieval_metrics()
            avg_losses.update(retrieval_metrics)
        
        self._log_evaluation_stats(avg_losses)
        return avg_losses
    
    def _compute_retrieval_metrics(self) -> Dict[str, float]:
        eval_dict = self.eval_data.to_dict()
        train_dict = self.train_data.to_dict()
        queries = eval_dict['query'][:self.config.max_eval_samples]
        ground_truth_docids = eval_dict['doc_id'][:self.config.max_eval_samples]
        # Use train_data DocIDs as valid set (this is what model learned)
        valid_docids = list(set(train_dict['doc_id']))
        
        # Set valid DocIDs for constrained generation
        self.model.set_valid_docids(valid_docids)
        
        metrics = self.model.evaluate_retrieval(
            queries=queries,
            ground_truth_docids=ground_truth_docids,
            valid_docids=valid_docids,
            k_values=self.config.eval_k_values
        )
        
        return {f'eval_{k}': v for k, v in metrics.items()}
    
    def train(self) -> Dict[str, List[float]]:
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.config.num_epochs}")
        self.logger.info(f"Steps per epoch: {len(self.train_dataloader)}")
        self.logger.info(f"Total steps: {len(self.train_dataloader) * self.config.num_epochs}")
        
        # Check if using two-phase training
        if getattr(self.config, 'use_two_phase_training', False):
            self.logger.info("🔄 Using two-phase DSI training strategy")
            return self._train_two_phase()
        
        history = {'train_loss': [], 'eval_loss': []}
        best_eval_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            
            eval_metrics = {}
            if self.eval_data is not None:
                eval_metrics = self.evaluate()
            
            all_metrics = {**train_metrics, **eval_metrics}
            self.metrics_tracker.log_epoch_metrics(epoch, all_metrics)
            
            history['train_loss'].append(train_metrics.get('train_total_loss', 0.0))
            if eval_metrics:
                history['eval_loss'].append(eval_metrics.get('eval_total_loss', 0.0))
            
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} completed")
            for key, value in all_metrics.items():
                self.logger.info(f"  {key}: {value:.4f}")
            
            if eval_metrics and eval_metrics.get('eval_total_loss', float('inf')) < best_eval_loss:
                best_eval_loss = eval_metrics['eval_total_loss']
                self._save_checkpoint(is_best=True)
            
            self._save_checkpoint()
        
        self.logger.info("Training completed!")
        return history
    
    def _train_two_phase(self) -> Dict[str, List[float]]:
        """Implement two-phase DSI training: indexing-only then fine-tuning with retrieval."""
        self.logger.info("Starting Two-Phase DSI Training")
        
        # Phase 1: Indexing-only training
        phase1_epochs = int(self.config.num_epochs * 0.6)  # 60% for indexing
        phase2_epochs = self.config.num_epochs - phase1_epochs  # 40% for both tasks
        
        self.logger.info(f"Phase 1: Indexing-only training ({phase1_epochs} epochs)")
        
        # Save original config
        original_task_type = self.config.task_type
        original_indexing_weight = self.config.indexing_weight
        original_retrieval_weight = self.config.retrieval_weight
        
        # Phase 1 configuration
        self.config.task_type = "indexing"
        self.config.indexing_weight = 1.0
        self.config.retrieval_weight = 0.0
        
        # Recreate dataloaders for indexing-only
        self._setup_dataloaders()
        
        history = {'train_loss': [], 'eval_loss': [], 'phase': []}
        best_eval_loss = float('inf')
        
        # Train Phase 1
        for epoch in range(phase1_epochs):
            self.current_epoch = epoch
            self.logger.info(f"Phase 1 - Epoch {epoch + 1}/{phase1_epochs}")
            
            train_metrics = self.train_epoch()
            
            eval_metrics = {}
            # Only evaluate every few epochs during indexing phase to avoid generation issues
            if self.eval_data is not None and (epoch + 1) % 3 == 0:
                try:
                    eval_metrics = self.evaluate()
                except Exception as e:
                    self.logger.warning(f"Evaluation failed at epoch {epoch + 1}: {e}")
                    eval_metrics = {'eval_total_loss': float('inf')}
            
            all_metrics = {**train_metrics, **eval_metrics}
            self.metrics_tracker.log_epoch_metrics(epoch, all_metrics)
            
            history['train_loss'].append(train_metrics.get('train_total_loss', 0.0))
            if eval_metrics:
                history['eval_loss'].append(eval_metrics.get('eval_total_loss', 0.0))
            history['phase'].append('indexing')
            
            # Save best model from Phase 1
            if eval_metrics and eval_metrics.get('eval_total_loss', float('inf')) < best_eval_loss:
                best_eval_loss = eval_metrics['eval_total_loss']
                self._save_checkpoint(is_best=True)
        
        self.logger.info(f"Phase 1 completed. Best eval loss: {best_eval_loss:.4f}")
        
        # Phase 2: Multi-task fine-tuning
        self.logger.info(f"Phase 2: Multi-task fine-tuning ({phase2_epochs} epochs)")
        
        # Phase 2 configuration
        self.config.task_type = "both"
        self.config.indexing_weight = original_indexing_weight * 0.5  # Reduce indexing weight
        self.config.retrieval_weight = original_retrieval_weight * 2.0  # Increase retrieval weight
        
        # Recreate dataloaders for both tasks
        self._setup_dataloaders()
        
        # Lower learning rate for Phase 2 (fine-tuning)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
        
        # Train Phase 2
        for epoch in range(phase2_epochs):
            self.current_epoch = phase1_epochs + epoch
            self.logger.info(f"Phase 2 - Epoch {epoch + 1}/{phase2_epochs}")
            
            train_metrics = self.train_epoch()
            
            eval_metrics = {}
            # Full evaluation during phase 2
            if self.eval_data is not None:
                try:
                    eval_metrics = self.evaluate()
                except Exception as e:
                    self.logger.warning(f"Evaluation failed at phase 2 epoch {epoch + 1}: {e}")
                    eval_metrics = {'eval_total_loss': float('inf')}
            
            all_metrics = {**train_metrics, **eval_metrics}
            self.metrics_tracker.log_epoch_metrics(self.current_epoch, all_metrics)
            
            history['train_loss'].append(train_metrics.get('train_total_loss', 0.0))
            if eval_metrics:
                history['eval_loss'].append(eval_metrics.get('eval_total_loss', 0.0))
            history['phase'].append('both')
            
            # Save best model from Phase 2
            if eval_metrics and eval_metrics.get('eval_total_loss', float('inf')) < best_eval_loss:
                best_eval_loss = eval_metrics['eval_total_loss']
                self._save_checkpoint(is_best=True)
        
        # Restore original config
        self.config.task_type = original_task_type
        self.config.indexing_weight = original_indexing_weight
        self.config.retrieval_weight = original_retrieval_weight
        
        self.logger.info(f"Two-phase training completed! Final best eval loss: {best_eval_loss:.4f}")
        
        return history
    
    def _log_training_stats(self, loss: float, learning_rate: float):
        self.logger.info(
            f"Step {self.global_step}: loss={loss:.4f}, lr={learning_rate:.2e}"
        )
    
    def _log_evaluation_stats(self, metrics: Dict[str, float]):
        self.logger.info("Evaluation results:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
    
    def _save_checkpoint(self, is_best: bool = False):
        if not self.config.output_dir:
            return
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'metrics': self.metrics_tracker.get_all_metrics()
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = os.path.join(
            self.config.output_dir, 
            f"checkpoint-step-{self.global_step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.config.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.model.save_model(os.path.join(self.config.output_dir, "best_model_hf"))
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        if 'metrics' in checkpoint:
            self.metrics_tracker.load_metrics(checkpoint['metrics'])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")