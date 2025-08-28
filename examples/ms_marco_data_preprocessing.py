"""
MS MARCO Data Preprocessing Example

This example demonstrates how to use the Generative Retrieval Data Library 
to preprocess the MS MARCO dataset for generative retrieval tasks.

Key capabilities demonstrated:
1. Data ingestion from Huggingface datasets
2. Schema validation and transformation
3. Default sequential doc_id generation
4. Custom doc_id generation methods
5. Full compatibility with Huggingface datasets functionality

To run this example (from the root directory): python -m examples.ms_marco_data_preprocessing
"""

from datasets import load_dataset
import pandas as pd
import hashlib

# Import the generative retrieval library
from generative_retrieval import GRData


# =============================================================================
# DATA TRANSFORMATION UTILITIES
# =============================================================================

def transform_ms_marco_to_schema2(example):
    """
    Transform MS MARCO dataset example to Schema 2 format (doc_text, query).
    
    MS MARCO structure:
    - query: the search query (string)
    - passages: dict with keys ['passage_text', 'is_selected', 'url'] 
                where each value is a list of passages
    
    Returns list of transformed examples (one per passage).
    """
    transformed_examples = []
    passage_texts = example['passages']['passage_text']
    
    for passage_text in passage_texts:
        transformed_examples.append({
            'doc_text': passage_text,
            'query': example['query']
        })
    
    return transformed_examples


def hash_doc_id_generator(example, idx):
    """
    Custom doc_id generator using MD5 hash of document text.
    
    Args:
        example: Dataset example containing doc_text
        idx: Index of the example in dataset
        
    Returns:
        Example with hash-based doc_id added
    """
    text_hash = hashlib.md5(example['doc_text'].encode()).hexdigest()[:12]
    example['doc_id'] = f"hash_{text_hash}"
    return example


# =============================================================================
# STEP 1: DATA INGESTION FROM HUGGINGFACE DATASETS
# =============================================================================

def demonstrate_data_ingestion():
    """Demonstrate loading and transforming MS MARCO dataset."""
    print("=" * 60)
    print("STEP 1: DATA INGESTION FROM HUGGINGFACE DATASETS")
    print("=" * 60)
    
    # Load MS MARCO dataset sample
    print("Loading MS MARCO dataset (sample: 1000 queries)...")
    raw_dataset = load_dataset("microsoft/ms_marco", 'v1.1', split="train[:1000]")
    
    print(f"✓ Loaded dataset with {len(raw_dataset)} queries")
    print(f"✓ Original columns: {raw_dataset.column_names}")
    
    # Transform raw data to Schema 2 format
    print("\nTransforming to Schema 2 format (doc_text, query)...")
    transformed_data = []
    for example in raw_dataset:
        transformed_examples = transform_ms_marco_to_schema2(example)
        transformed_data.extend(transformed_examples)
    
    print(f"✓ Transformed {len(raw_dataset)} queries into {len(transformed_data)} document-query pairs")
    
    # Show sample of transformed data
    sample_df = pd.DataFrame(transformed_data[:3])
    print(f"\nSample transformed data:")
    print(f"Columns: {sample_df.columns.tolist()}")
    for i, row in sample_df.iterrows():
        print(f"  Row {i}: query='{row['query']}', doc_text='{row['doc_text'][:50]}...'")
    
    return transformed_data


# =============================================================================
# STEP 2: GRDATA INGESTION WITH DEFAULT SEQUENTIAL DOC_ID
# =============================================================================

def demonstrate_default_doc_id_generation(transformed_data):
    """Demonstrate GRData ingestion with default sequential doc_id generation."""
    print("\n" + "=" * 60)
    print("STEP 2: GRDATA INGESTION WITH DEFAULT SEQUENTIAL DOC_ID")
    print("=" * 60)
    
    # Create GRData from transformed data using default sequential doc_id
    print("Creating GRData with default sequential doc_id generation...")
    data_dict = {
        'doc_text': [item['doc_text'] for item in transformed_data],
        'query': [item['query'] for item in transformed_data]
    }
    
    gr_data = GRData.from_dict(data_dict)  # Uses sequential by default
    
    # Validate schema and show results
    schema_info = gr_data.get_schema_info()
    print(f"✓ Created GRData with {len(gr_data)} rows")
    print(f"✓ Schema: {schema_info['schema_type']}")
    print(f"✓ Valid schema: {schema_info['is_valid']}")
    print(f"✓ Columns: {schema_info['columns']}")
    
    # Show sample with sequential doc_ids
    print(f"\nSample with sequential doc_ids:")
    for i in range(3):
        example = gr_data[i]
        print(f"  {example['doc_id']}: '{example['doc_text'][:50]}...'")
    
    return gr_data


# =============================================================================
# STEP 3: CUSTOM DOC_ID GENERATION METHODS
# =============================================================================

def demonstrate_custom_doc_id_generation():
    """Demonstrate custom doc_id generation using hash-based method."""
    print("\n" + "=" * 60)
    print("STEP 3: CUSTOM DOC_ID GENERATION METHODS")
    print("=" * 60)
    
    # Create sample data for testing custom doc_id generation
    sample_data = {
        'doc_text': [
            'Machine learning is a subset of artificial intelligence.',
            'Natural language processing enables computers to understand text.',
            'Deep learning uses neural networks with multiple layers.'
        ],
        'query': [
            'What is machine learning?',
            'How does NLP work?', 
            'Explain deep learning'
        ]
    }
    
    # Test 1: Default sequential method
    print("Testing default sequential doc_id generation:")
    gr_sequential = GRData.from_dict(sample_data)
    for i in range(3):
        example = gr_sequential[i]
        print(f"  {example['doc_id']}: '{example['doc_text'][:40]}...'")
    
    # Test 2: Custom hash-based method
    print(f"\nTesting custom hash-based doc_id generation:")
    gr_hashed = GRData.from_dict(sample_data, doc_id_method=hash_doc_id_generator)
    for i in range(3):
        example = gr_hashed[i]
        print(f"  {example['doc_id']}: '{example['doc_text'][:40]}...'")
    
    print(f"\n✓ Both methods successfully generated unique doc_ids")
    return gr_sequential, gr_hashed


# =============================================================================
# STEP 4: GRDATA FUNCTIONALITY DEMONSTRATION
# =============================================================================

def demonstrate_grdata_functionality(gr_data):
    """Demonstrate key GRData functionality and Huggingface compatibility."""
    print("\n" + "=" * 60)
    print("STEP 4: GRDATA FUNCTIONALITY DEMONSTRATION")
    print("=" * 60)
    
    print(f"Original dataset size: {len(gr_data)} rows")
    
    # Test filtering
    print("\nTesting filtering functionality:")
    filtered = gr_data.filter(lambda x: len(x['doc_text']) > 100)
    print(f"✓ Filtered to {len(filtered)} rows (doc_text > 100 characters)")
    
    # Test train/test split
    print(f"\nTesting train/test split:")
    split_data = gr_data.train_test_split(test_size=0.2, seed=42)
    print(f"✓ Train set: {len(split_data['train'])} rows")
    print(f"✓ Test set: {len(split_data['test'])} rows")
    
    # Test data selection
    print(f"\nTesting data selection:")
    sample_indices = [0, 100, 200]
    selected = gr_data.select(sample_indices)
    print(f"✓ Selected {len(selected)} specific rows")
    
    # Test conversion methods
    print(f"\nTesting data conversion:")
    pandas_df = gr_data.to_pandas()
    data_dict = gr_data.to_dict()
    print(f"✓ Converted to pandas DataFrame: {pandas_df.shape}")
    print(f"✓ Converted to dictionary: {len(data_dict['doc_id'])} entries")
    
    # Test streaming compatibility
    print(f"\nTesting streaming dataset compatibility:")
    streaming_dataset = load_dataset("microsoft/ms_marco", 'v1.1', split="train", streaming=True)
    sample_streaming = []
    for i, example in enumerate(streaming_dataset):
        if i >= 50:
            break
        transformed = transform_ms_marco_to_schema2(example)
        sample_streaming.extend(transformed)
    
    streaming_data_dict = {
        'doc_text': [item['doc_text'] for item in sample_streaming],
        'query': [item['query'] for item in sample_streaming]
    }
    gr_streaming = GRData.from_dict(streaming_data_dict)
    print(f"✓ Created GRData from streaming dataset: {len(gr_streaming)} rows")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute MS MARCO preprocessing example demonstrating all library capabilities."""
    print("🚀 MS MARCO DATA PREPROCESSING EXAMPLE")
    print("Demonstrating generative retrieval data preprocessing with the GR Data Library")
    
    # Step 1: Data ingestion and transformation
    transformed_data = demonstrate_data_ingestion()
    
    # Step 2: Default sequential doc_id generation  
    gr_data_main = demonstrate_default_doc_id_generation(transformed_data)
    
    # Step 3: Custom doc_id generation methods
    gr_sequential, gr_hashed = demonstrate_custom_doc_id_generation()
    
    # Step 4: GRData functionality demonstration
    demonstrate_grdata_functionality(gr_data_main)
    
    print("\n" + "=" * 60)
    print("✅ MS MARCO PREPROCESSING EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nThis example demonstrated:")
    print("• Loading and transforming MS MARCO dataset")
    print("• Converting to Schema 2 format (doc_text, query)")
    print("• Default sequential doc_id generation")
    print("• Custom hash-based doc_id generation")
    print("• Full Huggingface datasets compatibility")
    print("• Data filtering, splitting, and conversion operations")
    print("• Streaming dataset support")
    print("\nYou can now use these patterns to preprocess other datasets!")


if __name__ == "__main__":
    main()