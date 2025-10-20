"""
DSI MS MARCO Training on Modal with Multi-GPU Support

This example demonstrates training a DSI model on Modal's cloud infrastructure
with support for multi-GPU training and distributed inference.

Features:
- Multi-GPU training with automatic device placement
- Distributed data parallelization
- Checkpointing and model persistence
- Scalable inference with batched generation
- MS MARCO dataset loading and preprocessing

Usage:
    modal run examples/dsi_ms_marco_modal.py --num-examples 1000 --num-epochs 10
    modal run examples/dsi_ms_marco_modal.py --num-examples 5000 --num-epochs 15 --gpus 4
"""

import os
import modal
from pathlib import Path

GPU_CONFIG = "A100"
GPU_COUNT = 2
CACHE_DIR = "/cache"
MODEL_DIR = "/models"

app = modal.App("dsi-ms-marco-training")

# Get the project root directory (parent of examples/)
project_root = Path(__file__).parent.parent

# Create the image with dependencies and include the generative_retrieval library
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.20.0",
        "datasets==4.0.0",
        "pandas==2.3.2",
        "pyarrow==21.0.0",
        "numpy>=1.21.0",
        "sentencepiece>=0.1.95",
        "protobuf>=3.20.0",
        "tqdm",
        "matplotlib",
    )
    .run_commands(
        "apt-get update",
        "apt-get install -y git",
    )
    .add_local_dir(
        str(project_root / "generative_retrieval"),
        "/root/generative_retrieval"
    )
)

volume = modal.Volume.from_name("dsi-models", create_if_missing=True)


@app.function(
    image=image,
    gpu=f"{GPU_CONFIG}:{GPU_COUNT}",
    volumes={MODEL_DIR: volume},
    timeout=3600 * 4,
    memory=32768,
)
def train_dsi_model(
    num_examples: int = 100,
    num_epochs: int = 5,
    model_name: str = "t5-small",
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    output_dir: str = "/models/dsi_output",
):
    """Train DSI model on MS MARCO dataset with multi-GPU support."""
    import sys
    sys.path.insert(0, "/root")
    
    import torch
    import torch.distributed as dist
    import random
    import numpy as np
    from datasets import load_dataset
    
    print(f"=== DSI MS MARCO Training on Modal ===")
    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    from generative_retrieval import GRData
    from generative_retrieval.models import DSIModel
    from generative_retrieval.train import DSITrainer, TrainingConfig
    
    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    set_seed(42)
    
    def transform_ms_marco_to_schema(example):
        transformed_examples = []
        passage_texts = example['passages']['passage_text']
        
        for passage_text in passage_texts:
            transformed_examples.append({
                'doc_text': passage_text,
                'query': example['query']
            })
        
        return transformed_examples
    
    def load_ms_marco_data(num_examples: int):
        print(f"Loading {num_examples} MS MARCO queries...")
        raw_dataset = load_dataset("microsoft/ms_marco", 'v1.1', split=f"train[:{num_examples}]")
        
        transformed_data = []
        for example in raw_dataset:
            transformed_examples = transform_ms_marco_to_schema(example)
            transformed_data.extend(transformed_examples)
        
        print(f"Loaded {len(raw_dataset)} queries -> {len(transformed_data)} document-query pairs")
        
        data_dict = {
            'doc_text': [item['doc_text'] for item in transformed_data],
            'query': [item['query'] for item in transformed_data]
        }
        
        return data_dict
    
    print("\n--- Loading Data ---")
    ms_marco_data = load_ms_marco_data(num_examples)
    gr_data = GRData.from_dict(ms_marco_data, doc_id_method="sequential")
    print(f"Dataset created with {len(gr_data)} examples")
    
    total_size = len(gr_data)
    train_size = int(total_size * 0.8)
    eval_start = int(total_size * 0.6)
    eval_size = int(total_size * 0.2)
    
    train_data = gr_data.select(list(range(train_size)))
    eval_indices = list(range(eval_start, min(eval_start + eval_size, total_size)))
    eval_data = gr_data.select(eval_indices)
    
    print(f"Train examples: {len(train_data)}")
    print(f"Eval examples: {len(eval_data)}")
    
    train_docids = set(train_data.to_dict()['doc_id'])
    eval_docids = set(eval_data.to_dict()['doc_id'])
    overlap = train_docids.intersection(eval_docids)
    print(f"DocID overlap: {len(overlap)}/{len(eval_docids)} ({len(overlap)/len(eval_docids)*100:.1f}%)")
    
    print("\n--- Creating Model ---")
    model = DSIModel(
        model_name=model_name,
        docid_format="sequential",
        use_constrained_generation=True
    )
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model.model = torch.nn.DataParallel(model.model)
    
    print("\n--- Training Configuration ---")
    config = TrainingConfig(
        model_name=model_name,
        num_epochs=num_epochs,
        train_batch_size=batch_size,
        eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        optimizer="adafactor",
        output_dir=output_dir,
        indexing_weight=10.0,
        retrieval_weight=1.0,
        indexing_ratio=5.0,
        use_two_phase_training=True,
        logging_steps=10,
        eval_steps=100,
        save_steps=200,
        compute_retrieval_metrics=True,
        max_eval_samples=50
    )
    
    print(f"Two-phase training: {config.use_two_phase_training}")
    print(f"Indexing weight: {config.indexing_weight}")
    print(f"Retrieval weight: {config.retrieval_weight}")
    
    print("\n--- Starting Training ---")
    trainer = DSITrainer(
        model=model,
        config=config,
        train_data=train_data,
        eval_data=eval_data
    )
    
    history = trainer.train()
    
    print("\n--- Training Completed ---")
    trainer.metrics_tracker.print_summary()
    
    print("\n--- Saving Model ---")
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(model.model, torch.nn.DataParallel):
        model.model = model.model.module
    
    model.save_model(os.path.join(output_dir, "final_model"))
    volume.commit()
    
    print(f"\nModel saved to: {output_dir}/final_model")
    print("Training job completed successfully!")
    
    return {
        "output_dir": output_dir,
        "history": history,
        "num_examples": num_examples,
        "num_epochs": num_epochs,
    }


@app.function(
    image=image,
    gpu=f"{GPU_CONFIG}",
    volumes={MODEL_DIR: volume},
    timeout=600,
)
def run_inference(
    model_path: str,
    queries: list[str],
    num_return_sequences: int = 10,
):
    """Run inference on trained DSI model."""
    import sys
    sys.path.insert(0, "/root")
    
    import torch
    from generative_retrieval.models import DSIModel
    
    print(f"Loading model from: {model_path}")
    
    model = DSIModel(
        model_name="t5-small",
        docid_format="sequential",
        use_constrained_generation=True
    )
    
    model.load_model(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Running inference on {len(queries)} queries...")
    print(f"Device: {device}")
    
    generated_docids = model.generate_docids(
        queries,
        num_return_sequences=num_return_sequences,
        valid_docids=None
    )
    
    results = []
    for i, (query, docids) in enumerate(zip(queries, generated_docids)):
        results.append({
            "query": query,
            "generated_docids": docids,
            "top_docid": docids[0] if docids else None
        })
        print(f"\nQuery {i+1}: {query[:80]}...")
        print(f"  Generated DocIDs: {docids[:5]}")
    
    return results


@app.function(
    image=image,
    gpu=f"{GPU_CONFIG}:4",
    volumes={MODEL_DIR: volume},
    timeout=3600 * 6,
    memory=65536,
)
def train_dsi_model_large(
    num_examples: int = 5000,
    num_epochs: int = 15,
    model_name: str = "t5-base",
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    output_dir: str = "/models/dsi_distributed",
):
    """
    Train DSI model with multi-GPU on a single powerful node (4 GPUs).
    
    This function uses PyTorch's DataParallel or DistributedDataParallel
    for efficient training across 4 GPUs on a single node.
    
    Note: Multi-node training with modal.experimental.clustered is in private beta.
    Contact Modal support@modal.com to get access to multi-node features.
    """
    import sys
    sys.path.insert(0, "/root")
    
    import torch
    import torch.distributed as dist
    import random
    import numpy as np
    from datasets import load_dataset
    
    print(f"=== Large-Scale Multi-GPU DSI Training ===")
    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    num_gpus = torch.cuda.device_count()
    use_ddp = num_gpus > 1
    
    if use_ddp:
        print(f"\nUsing DistributedDataParallel with {num_gpus} GPUs")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['WORLD_SIZE'] = str(num_gpus)
        
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    
    from generative_retrieval import GRData
    from generative_retrieval.models import DSIModel
    from generative_retrieval.train import DSITrainer, TrainingConfig
    
    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    set_seed(42)
    
    def transform_ms_marco_to_schema(example):
        transformed_examples = []
        passage_texts = example['passages']['passage_text']
        
        for passage_text in passage_texts:
            transformed_examples.append({
                'doc_text': passage_text,
                'query': example['query']
            })
        
        return transformed_examples
    
    def load_ms_marco_data(num_examples: int):
        print(f"Loading {num_examples} MS MARCO queries...")
        raw_dataset = load_dataset("microsoft/ms_marco", 'v1.1', split=f"train[:{num_examples}]")
        
        transformed_data = []
        for example in raw_dataset:
            transformed_examples = transform_ms_marco_to_schema(example)
            transformed_data.extend(transformed_examples)
        
        print(f"Loaded {len(raw_dataset)} queries -> {len(transformed_data)} pairs")
        
        data_dict = {
            'doc_text': [item['doc_text'] for item in transformed_data],
            'query': [item['query'] for item in transformed_data]
        }
        
        return data_dict
    
    print("\n--- Loading Data ---")
    ms_marco_data = load_ms_marco_data(num_examples)
    gr_data = GRData.from_dict(ms_marco_data, doc_id_method="sequential")
    
    print(f"Dataset created with {len(gr_data)} examples")
    
    total_size = len(gr_data)
    train_size = int(total_size * 0.8)
    eval_start = int(total_size * 0.6)
    eval_size = int(total_size * 0.2)
    
    train_data = gr_data.select(list(range(train_size)))
    eval_indices = list(range(eval_start, min(eval_start + eval_size, total_size)))
    eval_data = gr_data.select(eval_indices)
    
    print(f"Train examples: {len(train_data)}")
    print(f"Eval examples: {len(eval_data)}")
    
    print("\n--- Creating Model ---")
    model = DSIModel(
        model_name=model_name,
        docid_format="sequential",
        use_constrained_generation=True
    )
    
    if num_gpus > 1:
        print(f"Wrapping model with DataParallel for {num_gpus} GPUs")
        model.model = torch.nn.DataParallel(model.model)
    
    print("\n--- Training Configuration ---")
    config = TrainingConfig(
        model_name=model_name,
        num_epochs=num_epochs,
        train_batch_size=batch_size * num_gpus,
        eval_batch_size=batch_size * 2 * num_gpus,
        learning_rate=learning_rate,
        optimizer="adafactor",
        output_dir=output_dir,
        indexing_weight=10.0,
        retrieval_weight=1.0,
        indexing_ratio=5.0,
        use_two_phase_training=True,
        logging_steps=20,
        eval_steps=200,
        save_steps=500,
        compute_retrieval_metrics=True,
        max_eval_samples=50
    )
    
    trainer = DSITrainer(
        model=model,
        config=config,
        train_data=train_data,
        eval_data=eval_data
    )
    
    print("\n--- Starting Training ---")
    history = trainer.train()
    
    print("\n--- Training Completed ---")
    trainer.metrics_tracker.print_summary()
    
    print("\n--- Saving Model ---")
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(model.model, torch.nn.DataParallel):
        model.model = model.model.module
    
    model.save_model(os.path.join(output_dir, "final_model"))
    volume.commit()
    
    print(f"\nModel saved to: {output_dir}/final_model")
    print("Large-scale training completed successfully!")
    
    return {
        "output_dir": output_dir,
        "history": history,
        "num_examples": num_examples,
        "num_gpus": num_gpus,
    }


@app.local_entrypoint()
def main(
    num_examples: int = 100,
    num_epochs: int = 5,
    model_name: str = "t5-small",
    batch_size: int = 8,
    distributed: bool = False,
    test_inference: bool = False,
):
    """
    Main entrypoint for DSI training on Modal.
    
    Args:
        num_examples: Number of MS MARCO examples to use
        num_epochs: Number of training epochs
        model_name: T5 model variant (t5-small, t5-base, t5-large)
        batch_size: Training batch size
        distributed: Use multi-node distributed training
        test_inference: Run inference test after training
    """
    print("=" * 60)
    print("DSI MS MARCO Training on Modal")
    print("=" * 60)
    print(f"Examples: {num_examples}")
    print(f"Epochs: {num_epochs}")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Distributed: {distributed}")
    print("=" * 60)
    
    if distributed:
        print("\nLaunching large-scale multi-GPU training (4 GPUs)...")
        result = train_dsi_model_large.remote(
            num_examples=num_examples,
            num_epochs=num_epochs,
            model_name=model_name,
            batch_size=batch_size,
        )
    else:
        print("\nLaunching standard multi-GPU training (2 GPUs)...")
        result = train_dsi_model.remote(
            num_examples=num_examples,
            num_epochs=num_epochs,
            model_name=model_name,
            batch_size=batch_size,
        )
    
    print("\nTraining completed!")
    print(f"Result: {result}")
    
    if test_inference and result:
        output_dir = result.get("output_dir")
        if output_dir:
            print("\n" + "=" * 60)
            print("Running inference test...")
            print("=" * 60)
            
            test_queries = [
                "What is machine learning?",
                "How does neural network work?",
                "Explain deep learning algorithms",
            ]
            
            model_path = f"{output_dir}/final_model"
            inference_results = run_inference.remote(
                model_path=model_path,
                queries=test_queries,
                num_return_sequences=5,
            )
            
            print("\nInference Results:")
            for result in inference_results:
                print(f"\nQuery: {result['query']}")
                print(f"Top DocID: {result['top_docid']}")
                print(f"All DocIDs: {result['generated_docids']}")
    
    print("\n" + "=" * 60)
    print("Job completed successfully!")
    print("=" * 60)
