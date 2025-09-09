"""
DSI MS MARCO Training Example

A minimal DSI training example using MS MARCO dataset.
Demonstrates the complete DSI training pipeline with a small data sample
for quick experimentation and testing.

This example demonstrates:
1. Loading MS MARCO dataset samples
2. Sequential DocID generation
3. DSI model training with proper two-phase approach
4. Evaluation and Hit@K metrics

Usage:
python -m examples.dsi_ms_marco
python -m examples.dsi_ms_marco --num_examples 100 --num_epochs 5
"""

import os
import torch
import random
import numpy as np
import argparse
from datasets import load_dataset

from generative_retrieval import GRData
from generative_retrieval.models import DSIModel
from generative_retrieval.train import DSITrainer, TrainingConfig, TrainingUtils


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def transform_ms_marco_to_schema2(example):
    """Transform MS MARCO example to doc_text, query format."""
    transformed_examples = []
    passage_texts = example['passages']['passage_text']
    
    for passage_text in passage_texts:
        transformed_examples.append({
            'doc_text': passage_text,
            'query': example['query']
        })
    
    return transformed_examples


def load_ms_marco_data(num_examples: int = 20):
    """Load a small MS MARCO sample for training."""
    print(f"Loading {num_examples} MS MARCO queries...")
    
    # Load small sample
    raw_dataset = load_dataset("microsoft/ms_marco", 'v1.1', split=f"train[:{num_examples}]")
    
    # Transform to schema format
    transformed_data = []
    for example in raw_dataset:
        transformed_examples = transform_ms_marco_to_schema2(example)
        transformed_data.extend(transformed_examples)
    
    print(f"Loaded {len(raw_dataset)} queries -> {len(transformed_data)} document-query pairs")
    
    # Convert to dictionary
    data_dict = {
        'doc_text': [item['doc_text'] for item in transformed_data],
        'query': [item['query'] for item in transformed_data]
    }
    
    return data_dict


def main():
    parser = argparse.ArgumentParser(description="DSI MS MARCO Training Example")
    parser.add_argument("--num_examples", type=int, default=20, 
                       help="Number of MS MARCO queries (default: 20)")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs (default: 3)")
    parser.add_argument("--model_name", type=str, default="t5-small",
                       help="Model name (default: t5-small)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size (default: 4)")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate (default: 1e-3)")
    parser.add_argument("--output_dir", type=str, default="./debug_output",
                       help="Output directory (default: ./debug_output)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    print("DSI MS MARCO TRAINING EXAMPLE")
    print("=" * 50)
    print(f"Examples: {args.num_examples}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print()
    
    # Set seed
    set_seed(args.seed)
    TrainingUtils.set_seed(args.seed)
    
    try:
        # Load data
        print("Loading data...")
        ms_marco_data = load_ms_marco_data(args.num_examples)
        gr_data = GRData.from_dict(ms_marco_data, doc_id_method="sequential")
        
        print(f"Dataset created with {len(gr_data)} examples")
        
        # Create overlapping train/eval sets to ensure DocID overlap
        # This is critical for DSI: eval set must contain DocIDs seen during training
        total_size = len(gr_data)
        train_size = int(total_size * 0.8)
        eval_start = int(total_size * 0.6)  # Start eval in the middle of train set
        eval_size = int(total_size * 0.2)
        
        # Train on first 80% of data
        train_data = gr_data.select(list(range(train_size)))
        
        # Eval on overlapping subset (60%-80% range) to ensure shared DocIDs
        eval_indices = list(range(eval_start, min(eval_start + eval_size, total_size)))
        eval_data = gr_data.select(eval_indices)
        
        print(f"Train examples: {len(train_data)}")
        print(f"Eval examples: {len(eval_data)}")
        
        # Verify DocID overlap (critical for DSI evaluation)
        train_docids = set(train_data.to_dict()['doc_id'])
        eval_docids = set(eval_data.to_dict()['doc_id'])
        overlap = train_docids.intersection(eval_docids)
        print(f"DocID overlap: {len(overlap)} out of {len(eval_docids)} eval DocIDs")
        print(f"Overlap percentage: {len(overlap)/len(eval_docids)*100:.1f}%")
        
        if len(overlap) == 0:
            print("WARNING: No DocID overlap! Model cannot hit eval targets.")
        elif len(overlap) / len(eval_docids) < 0.5:
            print("WARNING: Low DocID overlap! Performance will be limited.")
        else:
            print("Good: Sufficient DocID overlap for meaningful evaluation.")
        print()
        
        # Create model
        print("Creating DSI model...")
        model = DSIModel(
            model_name=args.model_name,
            docid_format="sequential",
            use_constrained_generation=True
        )
        
        total_params, trainable_params = TrainingUtils.count_parameters(model)
        print(f"Model created with {trainable_params:,} trainable parameters")
        print()
        
        # Setup training config for debugging
        print("Setting up training configuration...")
        config = TrainingConfig(
            model_name=args.model_name,
            num_epochs=args.num_epochs,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size * 2,
            learning_rate=args.learning_rate,
            optimizer="adafactor",
            output_dir=args.output_dir,
            
            # DSI-specific settings
            indexing_weight=10.0,
            retrieval_weight=1.0,
            indexing_ratio=5.0,
            use_two_phase_training=True,
            
            # Fast training settings
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            compute_retrieval_metrics=True,
            max_eval_samples=50
        )
        
        print("Training configuration:")
        print(f"  Two-phase training: {config.use_two_phase_training}")
        print(f"  Indexing weight: {config.indexing_weight}")
        print(f"  Retrieval weight: {config.retrieval_weight}")
        print()
        
        # Create trainer
        print("Creating trainer...")
        trainer = DSITrainer(
            model=model,
            config=config,
            train_data=train_data,
            eval_data=eval_data
        )
        
        # Start training
        print("Starting training...")
        history = trainer.train()
        
        print("\nTraining completed!")
        
        # Print metrics summary
        trainer.metrics_tracker.print_summary()
        
        # Test some sample queries
        print("\nTesting on sample queries:")
        eval_dict = eval_data.to_dict()
        train_dict = train_data.to_dict()
        test_queries = eval_dict['query'][:3]
        true_docids = eval_dict['doc_id'][:3]
        valid_docids = list(set(train_dict['doc_id']))
        
        print(f"Valid DocIDs for generation: {len(valid_docids)} total")
        print(f"Sample valid DocIDs: {sorted(valid_docids)[:5]}...{sorted(valid_docids)[-5:]}")
        print(f"True DocIDs to find: {true_docids}")
        print(f"True DocIDs in valid set: {[docid for docid in true_docids if docid in valid_docids]}")
        
        generated_docids = model.generate_docids(
            test_queries,
            num_return_sequences=3,
            valid_docids=valid_docids
        )
        
        for i, (query, true_docid, generated) in enumerate(zip(test_queries, true_docids, generated_docids)):
            hit = true_docid in generated
            in_valid_set = true_docid in valid_docids
            print(f"\nQuery {i+1}: {query[:80]}...")
            print(f"  True DocID: {true_docid} {'(in train set)' if in_valid_set else '(NOT in train set!)'}")
            print(f"  Generated: {generated}")
            print(f"  Hit: {'✓' if hit else '✗'}")
            if not in_valid_set:
                print(f"  Note: Target DocID was never seen during training!")
        
        # Calculate hit rate
        total_hits = sum(1 for i, (true_docid, generated) in enumerate(zip(true_docids, generated_docids)) if true_docid in generated)
        hit_rate = total_hits / len(true_docids) * 100
        trainable_targets = sum(1 for docid in true_docids if docid in valid_docids)
        
        print(f"\nSample Results Summary:")
        print(f"  Total queries tested: {len(true_docids)}")
        print(f"  Hits achieved: {total_hits}")
        print(f"  Hit rate: {hit_rate:.1f}%")
        print(f"  Trainable targets: {trainable_targets}/{len(true_docids)}")
        
        print(f"\nResults saved to: {config.output_dir}")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("- Try reducing --num_examples (e.g., 10)")
        print("- Try reducing --num_epochs (e.g., 2)")
        print("- Check that you have enough memory")


if __name__ == "__main__":
    main()