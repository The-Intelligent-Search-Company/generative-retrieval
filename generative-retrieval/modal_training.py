import modal
import os
from pathlib import Path

app = modal.App("dsi-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.20.0",
        "datasets>=4.0.0",
        "pandas>=2.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "sentencepiece>=0.1.95",
        "protobuf>=3.20.0",
        "tqdm",
        "pyarrow>=21.0.0"
    )
    .add_local_dir(
        Path(__file__).parent,
        remote_path="/root/generative_retrieval"
    )
)

volume = modal.Volume.from_name("dsi-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("dsi-checkpoints", create_if_missing=True)

TRAIN_DATA_PATH = "/data/train"
EVAL_DATA_PATH = "/data/eval"
CHECKPOINT_PATH = "/checkpoints"


@app.function(
    image=image,
    gpu="A100-40GB:2",
    volumes={
        "/data": volume,
        "/checkpoints": checkpoint_volume
    },
    timeout=3600
)
def preprocess_data(
    dataset_name: str = "ms_marco",
    max_docs: int = 10000,
    docid_strategy: str = "semantic",
    num_clusters: int = 100,
    queries_per_doc: int = 3
):
    import sys
    sys.path.append("/root/generative_retrieval")
    
    from data_preprocessing import DSIDatasetPreprocessor
    
    preprocessor = DSIDatasetPreprocessor(
        docid_strategy=docid_strategy,
        num_clusters=num_clusters,
        queries_per_doc=queries_per_doc
    )
    
    train_ds, test_ds = preprocessor.create_training_dataset(
        corpus_name=dataset_name,
        output_dir="/data",
        max_docs=max_docs
    )
    
    volume.commit()
    
    return {
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "message": "Data preprocessing completed successfully"
    }


@app.function(
    gpu="A100-40GB:2",
    image=image,
    volumes={
        "/data": volume,
        "/checkpoints": checkpoint_volume
    },
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")] if False else []
)
def train_dsi_test(
    model_name: str = "t5-small",
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
    phase_1_epochs: int = 2,
    phase_2_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    use_constrained_generation: bool = True
):
    import torch
    import torch.distributed as dist
    from transformers import set_seed
    import sys
    sys.path.append("/root/generative_retrieval")
    
    from model import DSIMultiTaskModel
    from trainer import DistributedDSITrainer, setup_distributed, cleanup_distributed
    
    import torch.multiprocessing as mp
    
    world_size = torch.cuda.device_count()
    rank = 0
    local_rank = 0
    
    set_seed(42)
    
    print(f"Starting test training with {world_size} A100 GPUs")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(world_size):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    import json
    docid_mapping_path = Path("/data/docid_mapping.jsonl")
    if docid_mapping_path.exists():
        with open(docid_mapping_path, "r") as f:
            valid_docids = [json.loads(line)["docid"] for line in f]
    else:
        valid_docids = None
        print("Warning: No docid_mapping.jsonl found, constrained generation disabled")
    
    model = DSIMultiTaskModel(
        model_name=model_name,
        valid_docids=valid_docids,
        use_constrained_generation=use_constrained_generation and valid_docids is not None,
        indexing_weight=1.0,
        retrieval_weight=1.0
    )
    
    trainer = DistributedDSITrainer(
        model=model,
        train_dataset_path=TRAIN_DATA_PATH,
        eval_dataset_path=EVAL_DATA_PATH,
        output_dir=CHECKPOINT_PATH,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        local_rank=-1,
        world_size=1,
        phase_1_epochs=phase_1_epochs,
        phase_2_epochs=phase_2_epochs,
        fp16=True,
        save_steps=100,
        eval_steps=50,
        logging_steps=10
    )
    
    trainer.train()
    
    checkpoint_volume.commit()
    print("Training completed and checkpoints saved!")
    
    return {
        "status": "completed",
        "message": f"Test training finished with {world_size} A100 GPUs"
    }


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/data": volume,
        "/checkpoints": checkpoint_volume
    },
    timeout=3600
)
def train_dsi_single_gpu(
    model_name: str = "t5-base",
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    num_epochs: int = 10,
    phase_1_epochs: int = 6,
    phase_2_epochs: int = 4,
    use_constrained_generation: bool = True
):
    import torch
    from transformers import set_seed
    import sys
    import json
    sys.path.append("/root/generative_retrieval")
    
    from model import DSIMultiTaskModel
    from trainer import DistributedDSITrainer
    
    set_seed(42)
    
    print(f"Starting single-GPU training")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    docid_mapping_path = Path("/data/docid_mapping.jsonl")
    if docid_mapping_path.exists():
        with open(docid_mapping_path, "r") as f:
            valid_docids = [json.loads(line)["docid"] for line in f]
    else:
        valid_docids = None
        print("Warning: No docid_mapping.jsonl found, constrained generation disabled")
    
    model = DSIMultiTaskModel(
        model_name=model_name,
        valid_docids=valid_docids,
        use_constrained_generation=use_constrained_generation and valid_docids is not None,
        indexing_weight=1.0,
        retrieval_weight=1.0
    )
    
    trainer = DistributedDSITrainer(
        model=model,
        train_dataset_path=TRAIN_DATA_PATH,
        eval_dataset_path=EVAL_DATA_PATH,
        output_dir=CHECKPOINT_PATH,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        local_rank=-1,
        world_size=1,
        phase_1_epochs=phase_1_epochs,
        phase_2_epochs=phase_2_epochs,
        fp16=True
    )
    
    trainer.train()
    
    checkpoint_volume.commit()
    print("Training completed and checkpoints saved!")
    
    return {
        "status": "completed",
        "message": "Single-GPU training finished successfully"
    }


@app.local_entrypoint()
def main(
    mode: str = "test",
    dataset_name: str = "squad",
    max_docs: int = 1000,
    model_name: str = "t5-small",
    batch_size: int = 16,
    num_epochs: int = 3
):
    if mode == "test":
        print("Running quick test: preprocessing + training on 2 A100s...")
        
        print("Step 1: Preprocessing small dataset...")
        preprocess_result = preprocess_data.remote(
            dataset_name=dataset_name,
            max_docs=max_docs,
            queries_per_doc=2
        )
        print(preprocess_result)
        
        print("\nStep 2: Training model on 2 A100s...")
        train_result = train_dsi_test.remote(
            model_name=model_name,
            batch_size=batch_size,
            num_epochs=num_epochs
        )
        print(train_result)
        
    elif mode == "preprocess":
        print(f"Starting data preprocessing for {dataset_name}...")
        result = preprocess_data.remote(
            dataset_name=dataset_name,
            max_docs=max_docs
        )
        print(result)
        
    elif mode == "train":
        if distributed:
            print("Starting distributed multi-node training...")
            result = train_dsi_distributed.remote(
                model_name=model_name,
                batch_size=batch_size,
                num_epochs=num_epochs
            )
        else:
            print("Starting single-GPU training...")
            result = train_dsi_single_gpu.remote(
                model_name=model_name,
                batch_size=batch_size,
                num_epochs=num_epochs
            )
        print(result)
        
    elif mode == "full":
        print("Running full pipeline: preprocessing + training...")
        
        print("Step 1: Preprocessing data...")
        preprocess_result = preprocess_data.remote(
            dataset_name=dataset_name,
            max_docs=max_docs
        )
        print(preprocess_result)
        
        print("\nStep 2: Training model...")
        if distributed:
            train_result = train_dsi_distributed.remote(
                model_name=model_name,
                batch_size=batch_size,
                num_epochs=num_epochs
            )
        else:
            train_result = train_dsi_single_gpu.remote(
                model_name=model_name,
                batch_size=batch_size,
                num_epochs=num_epochs
            )
        print(train_result)
        
    else:
        print(f"Unknown mode: {mode}. Use 'test', 'preprocess', 'train', or 'full'")
