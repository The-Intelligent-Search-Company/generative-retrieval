import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocIDGenerator:
    def __init__(self, strategy: str = "semantic", num_clusters: int = 100):
        self.strategy = strategy
        self.num_clusters = num_clusters
        self.encoder_model = None
        self.tokenizer = None
        
    def _initialize_encoder(self):
        if self.encoder_model is None:
            logger.info("Loading sentence encoder for semantic DocID generation...")
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.encoder_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
    def _get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        self._initialize_encoder()
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                   max_length=512, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.encoder_model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
                
        return np.vstack(embeddings)
    
    def generate_docids(self, documents: List[str]) -> List[str]:
        if self.strategy == "sequential":
            return [str(i) for i in range(len(documents))]
        
        elif self.strategy == "hierarchical":
            return [self._generate_hierarchical_id(i, len(documents)) for i in range(len(documents))]
        
        elif self.strategy == "semantic":
            logger.info("Generating semantic DocIDs via clustering...")
            embeddings = self._get_embeddings(documents)
            
            kmeans = KMeans(n_clusters=min(self.num_clusters, len(documents)), 
                          random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            cluster_counts = {}
            docids = []
            for cluster_id in cluster_labels:
                count = cluster_counts.get(cluster_id, 0)
                docid = f"{cluster_id:03d}_{count:04d}"
                docids.append(docid)
                cluster_counts[cluster_id] = count + 1
                
            return docids
        
        else:
            raise ValueError(f"Unknown DocID strategy: {self.strategy}")
    
    def _generate_hierarchical_id(self, idx: int, total: int, levels: int = 3) -> str:
        ids = []
        remaining = idx
        level_size = int(total ** (1.0 / levels)) + 1
        
        for _ in range(levels):
            ids.append(str(remaining % level_size))
            remaining //= level_size
            
        return "_".join(reversed(ids))


class DSIDatasetPreprocessor:
    def __init__(
        self, 
        docid_strategy: str = "semantic",
        num_clusters: int = 100,
        queries_per_doc: int = 5
    ):
        self.docid_generator = DocIDGenerator(strategy=docid_strategy, num_clusters=num_clusters)
        self.queries_per_doc = queries_per_doc
        
    def load_corpus(self, dataset_name: str, split: str = "train", 
                   text_column: str = "text", max_docs: Optional[int] = None) -> pd.DataFrame:
        logger.info(f"Loading dataset: {dataset_name}")
        
        if dataset_name == "wikipedia":
            dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split=split)
            texts = [item["text"][:500] if len(item["text"]) > 500 else item["text"] for item in dataset]
            df = pd.DataFrame({
                "doc_id_original": range(len(dataset)),
                "text": texts
            })
        elif dataset_name == "squad":
            dataset = load_dataset("rajpurkar/squad", split=split)
            texts = [item["context"] for item in dataset]
            df = pd.DataFrame({
                "doc_id_original": range(len(dataset)),
                "text": texts
            })
        elif dataset_name == "ms_marco":
            dataset = load_dataset("ms_marco", "v1.1", split=split)
            
            passages = []
            for item in dataset:
                if "passages" in item and "passage_text" in item["passages"]:
                    for i, passage_text in enumerate(item["passages"]["passage_text"]):
                        passages.append({
                            "doc_id_original": f"{item.get('query_id', len(passages))}_{i}",
                            "text": passage_text
                        })
                elif "query" in item:
                    passages.append({
                        "doc_id_original": item.get("query_id", len(passages)),
                        "text": item["query"]
                    })
                    
            df = pd.DataFrame(passages)
            
        elif dataset_name == "nq":
            dataset = load_dataset("natural_questions", split=split)
            df = pd.DataFrame({
                "doc_id_original": range(len(dataset)),
                "text": dataset["document"]["tokens"]
            })
        else:
            dataset = load_dataset(dataset_name, split=split)
            df = pd.DataFrame({
                "doc_id_original": range(len(dataset)),
                "text": dataset[text_column]
            })
            
        if max_docs:
            df = df.head(max_docs)
            
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        logger.info(f"Loaded {len(df)} documents")
        return df
    
    def generate_indexing_pairs(self, df: pd.DataFrame) -> Dataset:
        logger.info("Generating indexing training pairs (document → DocID)...")
        
        documents = df["text"].tolist()
        docids = self.docid_generator.generate_docids(documents)
        
        df["docid"] = docids
        
        indexing_data = {
            "input_text": documents,
            "target_docid": docids,
            "task_type": ["indexing"] * len(documents)
        }
        
        return Dataset.from_dict(indexing_data), df
    
    def generate_retrieval_pairs(
        self, 
        df: pd.DataFrame,
        query_dataset: Optional[str] = None
    ) -> Dataset:
        logger.info("Generating retrieval training pairs (query → DocID)...")
        
        retrieval_pairs = []
        
        if query_dataset:
            queries_df = self._load_queries(query_dataset)
            for _, row in queries_df.iterrows():
                if row["doc_id"] in df["doc_id_original"].values:
                    doc_row = df[df["doc_id_original"] == row["doc_id"]].iloc[0]
                    retrieval_pairs.append({
                        "input_text": row["query"],
                        "target_docid": doc_row["docid"],
                        "task_type": "retrieval"
                    })
        else:
            logger.info("No query dataset provided, generating synthetic queries...")
            retrieval_pairs = self._generate_synthetic_queries(df)
        
        logger.info(f"Generated {len(retrieval_pairs)} retrieval pairs")
        return Dataset.from_dict({k: [d[k] for d in retrieval_pairs] for k in retrieval_pairs[0].keys()})
    
    def _generate_synthetic_queries(self, df: pd.DataFrame) -> List[Dict]:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        logger.info("Initializing query generation model...")
        
        device = 0 if torch.cuda.is_available() else -1
        if device == -1:
            logger.warning("No GPU available for query generation, using CPU (this will be slower)")
        
        tokenizer = AutoTokenizer.from_pretrained("BeIR/query-gen-msmarco-t5-base-v1")
        model = AutoModelForSeq2SeqLM.from_pretrained("BeIR/query-gen-msmarco-t5-base-v1")
        
        if device == 0:
            model = model.cuda()
        
        retrieval_pairs = []
        for _, row in df.iterrows():
            doc_text = row["text"][:512]
            
            try:
                inputs = tokenizer(
                    f"Generate query: {doc_text}",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                )
                
                if device == 0:
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = model.generate(
                    **inputs,
                    max_length=64,
                    num_return_sequences=self.queries_per_doc,
                    num_beams=self.queries_per_doc + 2,
                    early_stopping=True
                )
                
                queries = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                
                for query in queries:
                    retrieval_pairs.append({
                        "input_text": query,
                        "target_docid": row["docid"],
                        "task_type": "retrieval"
                    })
            except Exception as e:
                logger.warning(f"Failed to generate queries for doc: {e}")
                continue
                
        return retrieval_pairs
    
    def _load_queries(self, query_dataset: str) -> pd.DataFrame:
        dataset = load_dataset(query_dataset)
        return pd.DataFrame(dataset)
    
    def create_training_dataset(
        self,
        corpus_name: str,
        output_dir: str,
        query_dataset: Optional[str] = None,
        max_docs: Optional[int] = None,
        train_split: float = 0.8
    ) -> Tuple[Dataset, Dataset]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df = self.load_corpus(corpus_name, max_docs=max_docs)
        
        indexing_dataset, df_with_docids = self.generate_indexing_pairs(df)
        retrieval_dataset = self.generate_retrieval_pairs(df_with_docids, query_dataset)
        
        df_with_docids.to_json(output_path / "docid_mapping.jsonl", orient="records", lines=True)
        
        from datasets import concatenate_datasets
        combined_dataset = concatenate_datasets([indexing_dataset, retrieval_dataset])
        
        combined_dataset = combined_dataset.shuffle(seed=42)
        split_datasets = combined_dataset.train_test_split(train_size=train_split, seed=42)
        
        split_datasets["train"].save_to_disk(str(output_path / "train"))
        split_datasets["test"].save_to_disk(str(output_path / "test"))
        
        logger.info(f"Training dataset size: {len(split_datasets['train'])}")
        logger.info(f"Test dataset size: {len(split_datasets['test'])}")
        logger.info(f"Saved to {output_path}")
        
        return split_datasets["train"], split_datasets["test"]


if __name__ == "__main__":
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
