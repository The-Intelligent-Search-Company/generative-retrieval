# DSI-Based Generative Retrieval with Modal

A production-ready implementation of Differentiable Search Index (DSI) for generative document retrieval, with distributed training support on Modal infrastructure.

## Features

- **T5-based DSI Model**: Encoder-decoder architecture for end-to-end document indexing and retrieval
- **Constrained Generation**: Trie-based beam search to ensure valid DocID generation
- **Multiple DocID Strategies**: Sequential, hierarchical, and semantic clustering-based DocIDs
- **Two-Phase Training**: Separate indexing and multi-task retrieval phases
- **Distributed Training**: Multi-node, multi-GPU training with PyTorch DDP on Modal
- **Comprehensive Evaluation**: Hit@K, MRR metrics with failure analysis

## Installation

```bash
pip install -r requirements.txt
```

For Modal deployment:
```bash
pip install modal
modal setup
```

## Quick Start

### 1. Data Preprocessing

```python
from generative_retrieval import DSIDatasetPreprocessor

preprocessor = DSIDatasetPreprocessor(
    docid_strategy="semantic",
    num_clusters=100,
    queries_per_doc=3
)

train_ds, test_ds = preprocessor.create_training_dataset(
    corpus_name="ms_marco",
    output_dir="./processed_data",
    max_docs=10000
)
```

### 2. Local Training (Single GPU)

```python
from generative_retrieval import DSIMultiTaskModel, DistributedDSITrainer

model = DSIMultiTaskModel(
    model_name="t5-base",
    use_constrained_generation=True
)

trainer = DistributedDSITrainer(
    model=model,
    train_dataset_path="./processed_data/train",
    eval_dataset_path="./processed_data/test",
    output_dir="./checkpoints",
    batch_size=32,
    num_epochs=10,
    phase_1_epochs=6,
    phase_2_epochs=4
)

trainer.train()
```

### 3. Modal Distributed Training

```bash
# Preprocess data
modal run generative_retrieval/modal_training.py --mode preprocess --dataset_name ms_marco --max_docs 10000

# Single-GPU training
modal run generative_retrieval/modal_training.py --mode train --distributed false

# Multi-node training (4 nodes x 8 H100s)
modal run generative_retrieval/modal_training.py --mode train --distributed true

# Full pipeline
modal run generative_retrieval/modal_training.py --mode full --dataset_name ms_marco --max_docs 50000
```

### 4. Evaluation

```python
from generative_retrieval import evaluate_checkpoint

metrics = evaluate_checkpoint(
    checkpoint_path="./checkpoints/best_model",
    eval_dataset_path="./processed_data/test",
    docid_mapping_path="./processed_data/docid_mapping.jsonl",
    output_path="./results.json",
    num_beams=10,
    k_values=[1, 5, 10]
)
```

## Architecture

### Model Architecture
```
Input Query/Document
      ↓
   T5 Encoder
      ↓
   T5 Decoder (with constrained generation)
      ↓
   DocID (string)
```

### Training Phases
1. **Phase 1 (60% epochs)**: Indexing-only training
   - Task: Document text → DocID
   - Objective: Memorize document-to-DocID mappings

2. **Phase 2 (40% epochs)**: Multi-task fine-tuning
   - Tasks: Document → DocID + Query → DocID
   - Objective: Learn retrieval while maintaining indexing

### DocID Strategies
- **Sequential**: `"0", "1", "2", ...`
- **Hierarchical**: `"0_0_1", "0_0_2", ...` (multi-level structure)
- **Semantic**: `"042_0003", "042_0004", ...` (cluster-based)

## Modal Infrastructure

### Multi-Node Configuration
```python
@app.function(
    gpu=modal.gpu.H100(count=8),  # 8 GPUs per node
    timeout=86400
)
@modal.experimental.clustered(size=4, rdma=True)  # 4 nodes
def train_dsi_distributed():
    # 32 total H100 GPUs with RDMA networking
    ...
```

### Resource Specifications
- **GPUs**: Up to 64 H100 SXM per training job
- **Network**: 50 Gbps IPv6 + 3200 Gbps RDMA
- **Storage**: Modal Volumes for datasets and checkpoints
- **Memory**: 1TB+ RAM per node

## Datasets Supported

- MS MARCO (default)
- Natural Questions (NQ)
- Custom datasets via HuggingFace `datasets` library

## Configuration

Key hyperparameters:
- `model_name`: T5 variant (`t5-base`, `t5-large`, `t5-xl`)
- `docid_strategy`: DocID generation method
- `batch_size`: Per-GPU batch size (scales with GPU count)
- `learning_rate`: 5e-5 recommended for T5-base
- `num_clusters`: For semantic DocID strategy
- `phase_1_epochs` / `phase_2_epochs`: Training phase split
- `indexing_weight` / `retrieval_weight`: Multi-task loss weights

## Performance

Expected results on MS MARCO 10K docs:
- **Hit@1**: 35-45%
- **Hit@10**: 60-75%
- **MRR**: 0.45-0.55
- **Training time**: ~2 hours (4 nodes x 8 H100s)

## Project Structure

```
generative-retrieval/
├── data_preprocessing.py   # Dataset loading and DocID generation
├── model.py               # DSI model and constrained generation
├── trainer.py             # Distributed training with PyTorch DDP
├── evaluation.py          # Hit@K metrics and failure analysis
├── modal_training.py      # Modal deployment functions
└── __init__.py           # Package exports
```

## Citation

Based on the DSI paper:
```
@article{tay2022transformer,
  title={Transformer memory as a differentiable search index},
  author={Tay, Yi and Tran, Vinh and Dehghani, Mostafa and Ni, Jianmo and Bahri, Dara and Mehta, Harsh and Qin, Zhen and Hui, Kai and Zhao, Zhe and Gupta, Jai and others},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## License

See LICENSE file for details.
