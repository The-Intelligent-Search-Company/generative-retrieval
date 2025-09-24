# Generative Retrieval Library

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*A production-ready library for generative retrieval systems and neural information retrieval*

</div>

## Overview

The Generative Retrieval Library provides a complete framework for training, evaluating, and deploying generative retrieval models. Unlike traditional retrieval systems that use separate indexing and ranking components, generative retrieval models directly generate document identifiers from queries using neural networks.

### Key Features

🚀 **Production Ready**: Battle-tested components with comprehensive error handling and logging  
🧩 **Modular Architecture**: Swappable datasets, encoders, ID schemes, and decoding backends  
⚡ **Optimized Training**: Multi-task learning, two-phase training, and advanced optimization  
📊 **Rich Evaluation**: Hit@K metrics, training curves, overfitting detection  
🎯 **Constrained Generation**: Trie-based constraints ensure valid document ID generation  
🔧 **Easy Integration**: Simple API with sensible defaults and extensive configuration options

## Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │    │   Model Layer    │    │ Training Layer  │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • GRData        │───▶│ • DSIModel       │───▶│ • DSITrainer    │
│ • DocID         │    │ • Tokenizers     │    │ • Config        │
│ • Preprocessing │    │ • Constraints    │    │ • Metrics       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

**Data Layer**: Unified data handling with automatic DocID generation and preprocessing  
**Model Layer**: Generative retrieval models with constrained generation and multiple DocID formats  
**Training Layer**: Advanced training strategies including two-phase learning and multi-task optimization

### Generative Retrieval Training Pipeline

1. **Indexing Phase**: Model learns `document content → document ID` mappings
2. **Retrieval Phase**: Model learns `query → document ID` mappings  
3. **Evaluation**: Generate DocIDs for queries and measure Hit@K performance

## Quick Start

### Installation

```bash
pip install torch transformers datasets
git clone https://github.com/The-Intelligent-Search-Company/generative-retrieval.git
cd generative-retrieval
pip install -e .
```

### Basic Usage

```python
from generative_retrieval import GRData, DSIModel, DSITrainer, ConfigPresets

# 1. Prepare your data
data = {
    'doc_text': ['Document content 1', 'Document content 2'],
    'query': ['Query about doc 1', 'Query about doc 2']
}
gr_data = GRData.from_dict(data)

# 2. Create and configure model
model = DSIModel(
    model_name="t5-base",
    docid_format="sequential",
    use_constrained_generation=True
)

# 3. Set up training configuration
config = ConfigPresets.dsi_two_phase()  # Optimized generative retrieval training

# 4. Train the model
trainer = DSITrainer(model=model, config=config, train_data=gr_data)
trainer.train()

# 5. Generate document IDs for queries
queries = ["What is machine learning?"]
generated_docids = model.generate_docids(queries, num_return_sequences=5)
```

### MS MARCO Example

```bash
# Quick test with 50 examples
python -m examples.dsi_ms_marco --num_examples 50 --num_epochs 3

# Production training with 5000 examples  
python -m examples.dsi_ms_marco --num_examples 5000 --num_epochs 15 --model_name t5-base
```

## API Documentation

### Core Classes

#### `GRData`
Unified data container for generative retrieval datasets.

```python
# Create from dictionary
gr_data = GRData.from_dict(data_dict)

# Create from files
gr_data = GRData.from_file("data.json", format="json")
gr_data = GRData.from_file("data.csv", format="csv")

# Train/test split
split_data = gr_data.train_test_split(test_size=0.2)
```

#### `DSIModel`
T5-based generative retrieval model with DSI implementation.

```python
model = DSIModel(
    model_name="t5-base",          # T5 model size
    docid_format="sequential",     # DocID format: "sequential", "hierarchical"
    max_docid_length=20,          # Maximum DocID length
    use_constrained_generation=True # Enable trie-based constraints
)

# Generate document IDs
docids = model.generate_docids(
    queries=["query text"],
    num_return_sequences=10,
    valid_docids=training_docids  # Optional: constrain to training set
)

# Evaluate retrieval performance  
metrics = model.evaluate_retrieval(queries, ground_truth_docids)
```

#### `DSITrainer`
Advanced trainer for generative retrieval with two-phase learning and multi-task optimization.

```python
trainer = DSITrainer(
    model=model,
    config=config,
    train_data=train_data,
    eval_data=eval_data  # Optional
)

# Train with automatic two-phase learning
history = trainer.train()

# Access training metrics
trainer.metrics_tracker.print_summary()
trainer.metrics_tracker.plot_metrics()
```

### Configuration Presets

Pre-configured training settings optimized for different use cases:

```python
# Quick testing (small model, few epochs)
config = ConfigPresets.quick_test()

# Development (balanced performance/speed)
config = ConfigPresets.development()

# Production (optimized for performance)
config = ConfigPresets.production()

# Two-phase generative retrieval training (recommended)
config = ConfigPresets.dsi_two_phase()

# Indexing-only training
config = ConfigPresets.indexing_only()

# Retrieval-only fine-tuning
config = ConfigPresets.retrieval_only()
```

### Advanced Configuration

```python
config = TrainingConfig(
    model_name="t5-base",
    num_epochs=25,
    train_batch_size=16,
    learning_rate=1e-3,
    optimizer="adafactor",
    
    # Generative retrieval settings
    indexing_weight=15.0,         # Weight for indexing loss
    retrieval_weight=1.0,         # Weight for retrieval loss  
    indexing_ratio=10.0,          # Data ratio (indexing:retrieval)
    use_two_phase_training=True,  # Enable two-phase learning
    
    # Training control
    use_mixed_precision=True,     # Enable AMP
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    
    # Evaluation
    compute_retrieval_metrics=True,
    eval_steps=200,
    save_steps=500
)
```

## Training Strategies

### Two-Phase Training (Recommended)

The library implements a two-phase training strategy optimized for generative retrieval:

1. **Phase 1 (60% of epochs)**: Indexing-only training
   - Task: `"index document: [text]" → "00000001"`  
   - Goal: Memorize document-to-DocID mappings
   
2. **Phase 2 (40% of epochs)**: Multi-task fine-tuning
   - Tasks: Both indexing and retrieval
   - Goal: Adapt memorized knowledge for query-based retrieval

### Multi-Task Loss Weighting

Control training emphasis with configurable loss weights:

```python
# Heavy indexing focus (recommended)
config.indexing_weight = 10.0
config.retrieval_weight = 1.0

# Balanced training
config.indexing_weight = 1.0  
config.retrieval_weight = 1.0
```

### Constrained Generation

Ensure generated DocIDs are valid using trie-based constraints:

```python
model = DSIModel(use_constrained_generation=True)

# Only generates DocIDs from the valid set
docids = model.generate_docids(queries, valid_docids=training_docids)
```

## Document ID Formats

### Sequential (Default)
Simple numeric identifiers: `00000001`, `00000002`, `00000003`...

```python
model = DSIModel(docid_format="sequential")
```

### Hierarchical
Structured identifiers for large collections: `100-10-5`, `100-10-6`...

```python  
model = DSIModel(docid_format="hierarchical")
```

### Semantic
Content-aware hierarchical clustering for better generalization.

```python
model = DSIModel(docid_format="semantic")
```

## Performance & Benchmarks

### Expected Performance

| Dataset Size | Model | Hit@1 | Hit@10 | Training Time |
|-------------|--------|-------|---------|---------------|
| 1K docs | T5-base | 60-80% | 85-95% | ~30 min |
| 10K docs | T5-base | 40-60% | 70-85% | ~3 hours |
| 100K docs | T5-large | 25-45% | 60-80% | ~24 hours |

### Optimization Tips

**For Small Datasets (< 5K docs):**
- Use `t5-small` or `t5-base`
- Set `indexing_weight=5.0`, `indexing_ratio=5.0`
- Enable `use_constrained_generation=True`

**For Large Datasets (50K+ docs):**
- Use `t5-large` or larger
- Set `indexing_weight=15.0`, `indexing_ratio=15.0`  
- Consider hierarchical DocID format
- Use gradient accumulation for larger effective batch sizes

## Examples and Tutorials

### Basic Training
```bash
# Train on MS MARCO subset
python -m examples.dsi_ms_marco --num_examples 100 --num_epochs 5
```

### Custom Dataset
```python
# Load your own data
data = {
    'doc_text': your_documents,
    'query': your_queries  
}
gr_data = GRData.from_dict(data)

# Train with custom config
config = TrainingConfig(
    model_name="t5-base",
    num_epochs=20,
    use_two_phase_training=True
)
```

### Evaluation and Metrics
```python
# Comprehensive evaluation
metrics = model.evaluate_retrieval(
    queries=test_queries,
    ground_truth_docids=true_docids,
    k_values=[1, 5, 10, 20]
)

print(f"Hit@1: {metrics['hits@1']:.2%}")
print(f"Hit@10: {metrics['hits@10']:.2%}")
```

## Development and Contributing

### Project Structure
```
generative_retrieval/
├── data/           # Data loading and preprocessing
├── models/         # DSI models and tokenizers  
├── train/          # Training utilities and configs
examples/           # Usage examples
```

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run example
python -m examples.dsi_ms_marco
```

## Citation

If you use this library in your research, please cite:

```bibtex
@misc{generative-retrieval-lib,
  title={Generative Retrieval},
  author={Asad Khan},
  year={2025},
  howpublished={\url{https://github.com/The-Intelligent-Search-Company/generative-retrieval}}
}
```

### Related Papers

- **DSI**: Tay et al. "Transformer Memory as a Differentiable Search Index" (NeurIPS 2022)
- **DSI-QG**: Zhuang et al. "Bridging the Gap Between Indexing and Retrieval for Differentiable Search Index with Query Generation" (2022)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

<div align="center">
  <strong>Built with ❤️ for the generative retrieval community</strong>
</div>
