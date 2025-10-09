import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import Dataset as TorchDataset
from transformers import get_linear_schedule_with_warmup
from datasets import load_from_disk
from typing import Optional, Dict, List
from pathlib import Path
import logging
import os
from tqdm import tqdm
import time

from .model import DSIMultiTaskModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DSIDataset(TorchDataset):
    def __init__(self, dataset, tokenizer, max_length: int = 512, max_target_length: int = 20):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        input_encoding = self.tokenizer(
            item["input_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            item["target_docid"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels,
            "task_type": item["task_type"]
        }


class DistributedDSITrainer:
    def __init__(
        self,
        model: DSIMultiTaskModel,
        train_dataset_path: str,
        eval_dataset_path: Optional[str] = None,
        output_dir: str = "./checkpoints",
        batch_size: int = 32,
        learning_rate: float = 5e-5,
        num_epochs: int = 10,
        warmup_steps: int = 1000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_steps: int = 1000,
        eval_steps: int = 500,
        logging_steps: int = 100,
        fp16: bool = True,
        local_rank: int = -1,
        world_size: int = 1,
        phase_1_epochs: int = 6,
        phase_2_epochs: int = 4
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.fp16 = fp16
        
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        
        self.phase_1_epochs = phase_1_epochs
        self.phase_2_epochs = phase_2_epochs
        
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading training dataset from {train_dataset_path}")
        train_dataset = load_from_disk(train_dataset_path)
        self.train_dataset = DSIDataset(train_dataset, model.dsi_model.tokenizer)
        
        if eval_dataset_path:
            logger.info(f"Loading evaluation dataset from {eval_dataset_path}")
            eval_dataset = load_from_disk(eval_dataset_path)
            self.eval_dataset = DSIDataset(eval_dataset, model.dsi_model.tokenizer)
        else:
            self.eval_dataset = None
            
        self.model.to(self.device)
        
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )
            
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        total_steps = (len(self.train_dataset) // (batch_size * world_size * gradient_accumulation_steps)) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
    def _get_dataloader(self, dataset: DSIDataset, shuffle: bool = True) -> DataLoader:
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=shuffle
            )
        else:
            sampler = None
            
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(shuffle and sampler is None),
            num_workers=4,
            pin_memory=True
        )
    
    def _train_epoch(self, dataloader: DataLoader, epoch: int, training_phase: str) -> float:
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        if self.is_distributed:
            dataloader.sampler.set_epoch(epoch)
            
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1} ({training_phase})",
            disable=self.local_rank not in [-1, 0]
        )
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            task_types = batch["task_type"]
            
            if training_phase == "phase_1_indexing" and any(t == "retrieval" for t in task_types):
                continue
            
            if self.fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        task_types=task_types
                    )
                    loss = outputs["loss"] / self.gradient_accumulation_steps
                    
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    task_types=task_types
                )
                loss = outputs["loss"] / self.gradient_accumulation_steps
                loss.backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                if self.global_step % self.logging_steps == 0 and self.local_rank in [-1, 0]:
                    logger.info(f"Step {self.global_step}, Loss: {loss.item() * self.gradient_accumulation_steps:.4f}")
                    
                if self.global_step % self.save_steps == 0 and self.local_rank in [-1, 0]:
                    self._save_checkpoint(f"checkpoint-step-{self.global_step}")
                    
                if self.eval_dataset and self.global_step % self.eval_steps == 0:
                    eval_loss = self._evaluate()
                    if self.local_rank in [-1, 0]:
                        logger.info(f"Evaluation loss: {eval_loss:.4f}")
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self._save_checkpoint("best_model")
                    self.model.train()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            progress_bar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})
            
        return total_loss / num_batches if num_batches > 0 else 0
    
    def _evaluate(self) -> float:
        if not self.eval_dataset:
            return 0.0
            
        self.model.eval()
        dataloader = self._get_dataloader(self.eval_dataset, shuffle=False)
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", disable=self.local_rank not in [-1, 0]):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                task_types = batch["task_type"]
                
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            task_types=task_types
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        task_types=task_types
                    )
                    
                total_loss += outputs["loss"].item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        if self.is_distributed:
            avg_loss_tensor = torch.tensor(avg_loss).to(self.device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / self.world_size
            
        return avg_loss
    
    def _save_checkpoint(self, checkpoint_name: str):
        save_path = self.output_dir / checkpoint_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_save.save_pretrained(str(save_path))
        
        torch.save({
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_eval_loss": self.best_eval_loss
        }, save_path / "trainer_state.pt")
        
        logger.info(f"Saved checkpoint to {save_path}")
    
    def train(self):
        logger.info(f"Starting training on rank {self.local_rank}")
        logger.info(f"Total epochs: {self.num_epochs} (Phase 1: {self.phase_1_epochs}, Phase 2: {self.phase_2_epochs})")
        
        dataloader = self._get_dataloader(self.train_dataset, shuffle=True)
        
        for epoch in range(self.num_epochs):
            if epoch < self.phase_1_epochs:
                training_phase = "phase_1_indexing"
                logger.info(f"Phase 1: Indexing-only training (Epoch {epoch+1}/{self.phase_1_epochs})")
            else:
                training_phase = "phase_2_multitask"
                logger.info(f"Phase 2: Multi-task training (Epoch {epoch+1-self.phase_1_epochs}/{self.phase_2_epochs})")
            
            epoch_loss = self._train_epoch(dataloader, epoch, training_phase)
            
            if self.local_rank in [-1, 0]:
                logger.info(f"Epoch {epoch+1} average loss: {epoch_loss:.4f}")
                
            if self.eval_dataset:
                eval_loss = self._evaluate()
                if self.local_rank in [-1, 0]:
                    logger.info(f"Epoch {epoch+1} evaluation loss: {eval_loss:.4f}")
                    
            if self.local_rank in [-1, 0]:
                self._save_checkpoint(f"checkpoint-epoch-{epoch+1}")
        
        if self.local_rank in [-1, 0]:
            self._save_checkpoint("final_model")
            logger.info("Training completed!")


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    logger.info(f"Initialized process group: rank {rank}, world_size {world_size}")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()
