"""
DSI Model and Tokenization Example

This example demonstrates DSI model loading and tokenization capabilities
using the Generative Retrieval Data Library with T5 backbone.

Key capabilities demonstrated:
1. Loading and preprocessing data for DSI
2. Creating DSI model with different docid formats
3. Tokenization for indexing and retrieval tasks
4. DocID encoding/decoding with different formats
5. Constrained generation setup
6. Integration with existing GRData infrastructure

To run this example (from the root directory): python -m examples.dsi_training_example
"""

import torch
from datasets import load_dataset
import pandas as pd
import hashlib
from typing import List, Dict, Any
import random
import numpy as np

from generative_retrieval import GRData
from generative_retrieval.models import DSIModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_sample_data():
    sample_data = {
        'doc_text': [
            'Machine learning is a subset of artificial intelligence that focuses on algorithms.',
            'Natural language processing enables computers to understand and generate human language.',
            'Deep learning uses neural networks with multiple layers for complex pattern recognition.',
            'Computer vision allows machines to interpret and understand visual information from images.',
            'Reinforcement learning trains agents through interaction with an environment using rewards.',
            'Data mining extracts useful patterns and knowledge from large datasets.',
            'Information retrieval systems help users find relevant documents from large collections.',
            'Search engines use complex algorithms to rank and retrieve relevant web pages.',
            'Neural networks are inspired by biological neurons and process information in layers.',
            'Knowledge graphs represent relationships between entities in structured formats.'
        ],
        'query': [
            'What is machine learning?',
            'How does NLP work with computers?',
            'Explain deep learning neural networks',
            'What is computer vision used for?',
            'How does reinforcement learning work?',
            'What is data mining?',
            'How do information retrieval systems work?',
            'How do search engines rank pages?',
            'What are neural networks?',
            'What are knowledge graphs?'
        ]
    }
    return sample_data


def semantic_docid_generator(example, idx):
    text_hash = hashlib.md5(example['doc_text'].encode()).hexdigest()
    hash_int = int(text_hash[:8], 16)
    
    level_1 = hash_int % 10
    level_2 = (hash_int // 10) % 5
    level_3 = (hash_int // 50) % 3
    
    example['doc_id'] = f"{level_1}-{level_2}-{level_3}"
    return example


def demonstrate_data_preparation():
    print("=" * 60)
    print("STEP 1: DATA PREPARATION FOR DSI TRAINING")
    print("=" * 60)
    
    print("Creating sample dataset...")
    sample_data = create_sample_data()
    
    print("Testing different docid generation methods:")
    
    print("\n1. Sequential DocIDs:")
    gr_sequential = GRData.from_dict(sample_data)
    for i in range(3):
        example = gr_sequential[i]
        print(f"  {example['doc_id']}: '{example['doc_text'][:50]}...'")
    
    print("\n2. Semantic Hierarchical DocIDs:")
    gr_semantic = GRData.from_dict(sample_data, doc_id_method=semantic_docid_generator)
    for i in range(3):
        example = gr_semantic[i]
        print(f"  {example['doc_id']}: '{example['doc_text'][:50]}...'")
    
    print(f"\n✓ Created datasets with {len(gr_sequential)} examples each")
    return gr_sequential, gr_semantic


def demonstrate_dsi_model_creation():
    print("\n" + "=" * 60)
    print("STEP 2: DSI MODEL CREATION AND SETUP")
    print("=" * 60)
    
    print("Creating DSI models with different configurations:")
    
    print("\n1. Sequential DocID DSI Model:")
    model_sequential = DSIModel(
        model_name="t5-small",
        docid_format="sequential",
        max_docid_length=10,
        use_constrained_generation=True
    )
    print(f"  ✓ Model size: {model_sequential.get_model_size():,} parameters")
    print(f"  ✓ Max docid length: {model_sequential.max_docid_length}")
    
    print("\n2. Hierarchical DocID DSI Model:")
    model_hierarchical = DSIModel(
        model_name="t5-small",
        docid_format="hierarchical",
        max_docid_length=15,
        use_constrained_generation=True
    )
    print(f"  ✓ Model size: {model_hierarchical.get_model_size():,} parameters")
    print(f"  ✓ Docid format: {model_hierarchical.docid_format}")
    
    return model_sequential, model_hierarchical


def demonstrate_tokenization(gr_data: GRData, model: DSIModel):
    print("\n" + "=" * 60)
    print("STEP 3: TOKENIZATION DEMONSTRATION")
    print("=" * 60)
    
    data_dict = gr_data.to_dict()
    doc_texts = data_dict['doc_text']
    queries = data_dict['query']
    doc_ids = data_dict['doc_id']
    
    print(f"Data overview:")
    print(f"  Documents: {len(doc_texts)}")
    print(f"  Queries: {len(queries)}")
    print(f"  DocIDs: {len(doc_ids)}")
    
    print("\nTokenization examples:")
    
    print("\n1. Document Tokenization (for indexing):")
    sample_doc = doc_texts[0]
    sample_docid = doc_ids[0]
    
    indexing_input = f"index document: {sample_doc}"
    input_tokens = model.tokenizer.encode(indexing_input)
    target_tokens = model.tokenizer.encode(sample_docid)
    
    print(f"  Input text: '{indexing_input[:60]}...'")
    print(f"  Input tokens: {input_tokens[:10]}... (length: {len(input_tokens)})")
    print(f"  Target docid: '{sample_docid}'")
    print(f"  Target tokens: {target_tokens}")
    
    print("\n2. Query Tokenization (for retrieval):")
    sample_query = queries[0]
    
    retrieval_input = f"retrieve query: {sample_query}"
    input_tokens = model.tokenizer.encode(retrieval_input)
    
    print(f"  Input text: '{retrieval_input}'")
    print(f"  Input tokens: {input_tokens}")
    print(f"  Target docid: '{sample_docid}'")
    print(f"  Target tokens: {target_tokens}")
    
    print("\n3. DocID Tokenizer Testing:")
    docid_tokenizer = model.docid_tokenizer
    
    for i, docid in enumerate(doc_ids[:3]):
        encoded = docid_tokenizer.encode_docid(docid)
        decoded = docid_tokenizer.decode_docid(encoded)
        is_valid = docid_tokenizer.validate_docid(docid)
        
        print(f"  DocID {i+1}: '{docid}'")
        print(f"    Encoded: {encoded}")
        print(f"    Decoded: '{decoded}'")
        print(f"    Valid: {is_valid}")
    
    print("\n4. Batch Processing:")
    indexing_batch = model.prepare_indexing_batch(doc_texts[:3], doc_ids[:3])
    retrieval_batch = model.prepare_retrieval_batch(queries[:3], doc_ids[:3])
    
    print(f"  Indexing batch - Input shape: {indexing_batch['input_ids'].shape}")
    print(f"  Indexing batch - Label shape: {indexing_batch['labels'].shape}")
    print(f"  Retrieval batch - Input shape: {retrieval_batch['input_ids'].shape}")
    print(f"  Retrieval batch - Label shape: {retrieval_batch['labels'].shape}")
    
    return doc_texts, queries, doc_ids


def demonstrate_generation_setup(model: DSIModel, queries: List[str], doc_ids: List[str]):
    print("\n" + "=" * 60)
    print("STEP 4: GENERATION SETUP AND TESTING")
    print("=" * 60)
    
    print("Setting up constrained generation:")
    model.set_valid_docids(doc_ids)
    print(f"✓ Configured {len(doc_ids)} valid docids for constraints")
    
    print("\nTesting generation capabilities (untrained model):")
    test_queries = queries[:3]
    
    print("Sample generation inputs:")
    for i, query in enumerate(test_queries):
        formatted_input = f"retrieve query: {query}"
        tokens = model.tokenizer.encode(formatted_input)
        print(f"  Query {i+1}: '{query}'")
        print(f"    Formatted: '{formatted_input}'")
        print(f"    Token count: {len(tokens)}")
        print(f"    Expected docid: '{doc_ids[i]}'")
    
    print("\nNote: Actual generation requires trained model weights")
    print("This setup is ready for training implementation")


def demonstrate_constraint_validation(model: DSIModel, doc_ids: List[str]):
    print("\n" + "=" * 60)
    print("STEP 5: CONSTRAINT VALIDATION")
    print("=" * 60)
    
    print("Testing constraint system:")
    
    if model.trie_constraint:
        print("✓ Trie constraint system initialized")
        
        print("\nValid docids in system:")
        for i, docid in enumerate(doc_ids[:5]):
            is_valid = model.trie_constraint.is_valid_docid(docid)
            print(f"  {docid}: {'✓ Valid' if is_valid else '✗ Invalid'}")
        
        print("\nTesting invalid docids:")
        invalid_docids = ["99999999", "invalid-docid", "wrong-format"]
        for docid in invalid_docids:
            is_valid = model.trie_constraint.is_valid_docid(docid)
            print(f"  {docid}: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    else:
        print("No constraint system enabled")


def demonstrate_data_split_and_streaming():
    print("\n" + "=" * 60)
    print("STEP 6: DATA SPLITTING FOR TRAINING")
    print("=" * 60)
    
    sample_data = create_sample_data()
    gr_data = GRData.from_dict(sample_data)
    
    print("Testing train/test split:")
    split_data = gr_data.train_test_split(test_size=0.3, seed=42)
    print(f"  Train set: {len(split_data['train'])} samples")
    print(f"  Test set: {len(split_data['test'])} samples")
    
    print("\nData ready for training pipeline:")
    train_data = split_data['train']
    test_data = split_data['test']
    
    train_dict = train_data.to_dict()
    test_dict = test_data.to_dict()
    
    print(f"  Training docids: {train_dict['doc_id']}")
    print(f"  Test docids: {test_dict['doc_id']}")
    
    print("✓ Data splitting ready for generative_retrieval/training module")


def main():
    print("🚀 DSI MODEL AND TOKENIZATION EXAMPLE")
    print("Demonstrating DSI model loading and tokenization with the Generative Retrieval Library")
    
    set_seed(42)
    
    try:
        gr_sequential, gr_semantic = demonstrate_data_preparation()
        
        model_sequential, model_hierarchical = demonstrate_dsi_model_creation()
        
        doc_texts, queries, doc_ids = demonstrate_tokenization(gr_sequential, model_sequential)
        
        demonstrate_generation_setup(model_sequential, queries, doc_ids)
        
        demonstrate_constraint_validation(model_sequential, doc_ids)
        
        demonstrate_data_split_and_streaming()
        
        print("\n" + "=" * 60)
        print("✅ DSI MODEL AND TOKENIZATION EXAMPLE COMPLETED!")
        print("=" * 60)
        print("\nThis example demonstrated:")
        print("• DSI model creation and loading with T5 backbone")
        print("• Tokenization for indexing and retrieval tasks")
        print("• Sequential and hierarchical docid formats")
        print("• DocID encoding/decoding with custom tokenizer")
        print("• Constrained generation setup with Trie structure")
        print("• Batch preparation for training")
        print("• Integration with existing GRData infrastructure")
        print("• Data splitting ready for training pipeline")
        print("\nReady for training implementation in:")
        print("• generative_retrieval/training/ module")
        print("• Actual training loops with optimizers")
        print("• Multi-task training (indexing + retrieval)")
        print("• Evaluation and metrics tracking")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {str(e)}")
        print("This might be due to missing dependencies or model download issues.")
        print("Make sure you have all required dependencies installed:")
        print("pip install transformers torch datasets sentencepiece protobuf")
        print("\nIf you're still having issues, try:")
        print("pip install --upgrade transformers sentencepiece")
        print("pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("\nThis example focuses on model loading and tokenization.")
        print("Training implementation will be in generative_retrieval/training/")


if __name__ == "__main__":
    main()