from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
import os


@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = "t5-base"
    docid_format: str = "sequential"
    max_docid_length: int = 20
    use_constrained_generation: bool = True
    
    # Training configuration
    num_epochs: int = 10
    train_batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    
    # Task configuration
    task_type: str = "both"  # "indexing", "retrieval", or "both"
    max_input_length: int = 512
    max_target_length: int = 20
    
    # Optimization configuration
    optimizer: str = "adafactor"  # "adamw", "adam", or "adafactor"
    scheduler: str = "linear"  # "linear", "cosine", or None
    use_mixed_precision: bool = True
    use_gpu: bool = True
    
    # Logging and evaluation
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    compute_retrieval_metrics: bool = True
    max_eval_samples: int = 1000
    eval_k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    
    # I/O configuration
    output_dir: Optional[str] = None
    save_total_limit: int = 3
    num_workers: int = 0
    
    # Data configuration
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    doc_id_method: str = "sequential"
    
    # Advanced training options
    label_smoothing: float = 0.0
    dropout_rate: float = 0.1
    seed: int = 42
    
    # Multi-task training weights
    indexing_weight: float = 10.0
    retrieval_weight: float = 1.0
    
    # DSI-specific configuration
    indexing_ratio: float = 10.0  # Ratio of indexing to retrieval examples
    use_two_phase_training: bool = True  # Enable two-phase training strategy
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = f"./outputs/dsi_{self.model_name.replace('/', '_')}"
        
        if self.task_type not in ["indexing", "retrieval", "both"]:
            raise ValueError(f"Invalid task_type: {self.task_type}")
        
        if self.optimizer not in ["adamw", "adam", "adafactor"]:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")
        
        if self.scheduler not in ["linear", "cosine", None]:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TrainingConfig':
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def to_json(self, json_path: str):
        os.makedirs(os.path.dirname(json_path) if os.path.dirname(json_path) else '.', exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config key '{key}' ignored")
    
    def get_effective_batch_size(self) -> int:
        return self.train_batch_size * self.gradient_accumulation_steps
    
    def get_total_steps(self, num_training_examples: int) -> int:
        steps_per_epoch = num_training_examples // self.get_effective_batch_size()
        return steps_per_epoch * self.num_epochs
    
    def get_warmup_steps(self, num_training_examples: int) -> int:
        total_steps = self.get_total_steps(num_training_examples)
        return int(total_steps * self.warmup_ratio)
    
    def validate(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        
        if self.max_input_length <= 0:
            raise ValueError("max_input_length must be positive")
        
        if self.max_target_length <= 0:
            raise ValueError("max_target_length must be positive")
        
        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must be between 0 and 1")
        
        if not self.eval_k_values:
            raise ValueError("eval_k_values cannot be empty")
        
        if any(k <= 0 for k in self.eval_k_values):
            raise ValueError("All values in eval_k_values must be positive")
    
    def __str__(self) -> str:
        return f"TrainingConfig(\n" + "\n".join([
            f"  {key}: {value}" for key, value in self.to_dict().items()
        ]) + "\n)"


# Predefined configurations for common use cases
class ConfigPresets:
    
    @staticmethod
    def quick_test() -> TrainingConfig:
        return TrainingConfig(
            model_name="t5-small",
            num_epochs=5,
            train_batch_size=8,
            eval_batch_size=16,
            learning_rate=1e-3,
            indexing_weight=10.0,
            retrieval_weight=1.0,
            indexing_ratio=5.0,
            max_eval_samples=100,
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            use_two_phase_training=True
        )
    
    @staticmethod
    def development() -> TrainingConfig:
        return TrainingConfig(
            model_name="t5-base",
            num_epochs=15,
            train_batch_size=16,
            eval_batch_size=32,
            learning_rate=1e-3,
            indexing_weight=10.0,
            retrieval_weight=1.0,
            indexing_ratio=10.0,
            max_eval_samples=500,
            use_two_phase_training=True
        )
    
    @staticmethod
    def production() -> TrainingConfig:
        return TrainingConfig(
            model_name="t5-large",
            num_epochs=30,
            train_batch_size=32,
            eval_batch_size=64,
            gradient_accumulation_steps=2,
            learning_rate=1e-3,
            weight_decay=0.01,
            indexing_weight=15.0,
            retrieval_weight=1.0,
            indexing_ratio=15.0,
            use_mixed_precision=True,
            compute_retrieval_metrics=True,
            use_two_phase_training=True,
            optimizer="adafactor"
        )
    
    @staticmethod
    def large_scale() -> TrainingConfig:
        return TrainingConfig(
            model_name="t5-xl",
            num_epochs=20,
            train_batch_size=8,
            eval_batch_size=16,
            gradient_accumulation_steps=8,
            learning_rate=5e-6,
            weight_decay=0.01,
            use_mixed_precision=True,
            max_grad_norm=0.5,
            eval_steps=1000,
            save_steps=2000,
            compute_retrieval_metrics=True,
            max_eval_samples=2000
        )
    
    @staticmethod
    def indexing_only() -> TrainingConfig:
        return TrainingConfig(
            model_name="t5-base",
            task_type="indexing",
            num_epochs=20,
            train_batch_size=32,
            learning_rate=1e-3,
            retrieval_weight=0.0,
            indexing_weight=1.0,
            indexing_ratio=1.0,
            training_phase="indexing",
            use_two_phase_training=False
        )
    
    @staticmethod
    def retrieval_only() -> TrainingConfig:
        return TrainingConfig(
            model_name="t5-base",
            task_type="retrieval",
            num_epochs=10,
            train_batch_size=32,
            learning_rate=5e-4,
            indexing_weight=0.0,
            retrieval_weight=1.0,
            indexing_ratio=0.0,
            training_phase="retrieval",
            use_two_phase_training=False,
            compute_retrieval_metrics=True
        )
    
    @staticmethod
    def hierarchical_docids() -> TrainingConfig:
        return TrainingConfig(
            model_name="t5-base",
            docid_format="hierarchical",
            max_docid_length=30,
            num_epochs=5,
            train_batch_size=12,
            learning_rate=3e-5
        )
    
    @staticmethod
    def dsi_two_phase() -> TrainingConfig:
        """Optimized configuration for two-phase DSI training.
        
        This preset implements the recommended DSI training strategy:
        - Phase 1: Indexing-only training for document memorization
        - Phase 2: Multi-task fine-tuning with both indexing and retrieval
        """
        return TrainingConfig(
            model_name="t5-base",
            num_epochs=25,
            train_batch_size=16,
            eval_batch_size=32,
            learning_rate=1e-3,
            indexing_weight=15.0,
            retrieval_weight=1.0,
            indexing_ratio=15.0,
            use_two_phase_training=True,
            optimizer="adafactor",
            compute_retrieval_metrics=True,
            logging_steps=50,
            eval_steps=200,
            save_steps=500
        )
    
    @staticmethod
    def get_preset(preset_name: str) -> TrainingConfig:
        presets = {
            "quick_test": ConfigPresets.quick_test,
            "development": ConfigPresets.development,
            "production": ConfigPresets.production,
            "large_scale": ConfigPresets.large_scale,
            "indexing_only": ConfigPresets.indexing_only,
            "retrieval_only": ConfigPresets.retrieval_only,
            "hierarchical_docids": ConfigPresets.hierarchical_docids,
            "dsi_two_phase": ConfigPresets.dsi_two_phase
        }
        
        if preset_name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
        
        return presets[preset_name]()
    
    @staticmethod
    def list_presets() -> List[str]:
        return [
            "quick_test",
            "development", 
            "production",
            "large_scale",
            "indexing_only",
            "retrieval_only",
            "hierarchical_docids",
            "dsi_two_phase"
        ]