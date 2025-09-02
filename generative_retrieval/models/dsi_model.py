from typing import List, Dict, Any, Optional, Union
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers.generation.utils import BeamSearchScorer
from .constraints import TrieConstraint
from .tokenizers import DocIDTokenizer


class DSIModel(nn.Module):
    def __init__(
        self,
        model_name: str = "t5-large",
        docid_format: str = "sequential",
        max_docid_length: int = 20,
        use_constrained_generation: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        self.docid_format = docid_format
        self.max_docid_length = max_docid_length
        self.use_constrained_generation = use_constrained_generation
        
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.docid_tokenizer = DocIDTokenizer(self.tokenizer, docid_format)
        
        self.trie_constraint = None
        if use_constrained_generation:
            self.trie_constraint = TrieConstraint(self.tokenizer)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def prepare_indexing_batch(self, doc_texts: List[str], doc_ids: List[str]) -> Dict[str, torch.Tensor]:
        inputs = [f"index document: {text}" for text in doc_texts]
        targets = doc_ids
        
        input_encodings = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        target_encodings = self.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.max_docid_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    
    def prepare_retrieval_batch(self, queries: List[str], doc_ids: List[str]) -> Dict[str, torch.Tensor]:
        inputs = [f"retrieve query: {query}" for query in queries]
        targets = doc_ids
        
        input_encodings = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        target_encodings = self.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.max_docid_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    
    def generate_docids(
        self,
        queries: List[str],
        num_return_sequences: int = 10,
        valid_docids: Optional[List[str]] = None,
        do_sample: bool = False,
        temperature: float = 1.0
    ) -> List[List[str]]:
        inputs = [f"retrieve query: {query}" for query in queries]
        
        input_encodings = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        generation_kwargs = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'max_length': self.max_docid_length,
            'num_return_sequences': num_return_sequences,
            'do_sample': do_sample,
            'temperature': temperature,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'return_dict_in_generate': True,
            'output_scores': True
        }
        
        if self.use_constrained_generation and valid_docids is not None:
            self.trie_constraint.build_trie(valid_docids)
            generation_kwargs['constraints'] = [self.trie_constraint.get_constraint()]
        
        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)
        
        generated_sequences = outputs.sequences
        batch_size = len(queries)
        
        results = []
        for i in range(batch_size):
            batch_results = []
            start_idx = i * num_return_sequences
            end_idx = (i + 1) * num_return_sequences
            
            for j in range(start_idx, min(end_idx, len(generated_sequences))):
                docid = self.tokenizer.decode(generated_sequences[j], skip_special_tokens=True)
                docid = docid.strip()
                if docid:
                    batch_results.append(docid)
            
            results.append(batch_results)
        
        return results
    
    def index_documents(self, doc_texts: List[str], doc_ids: List[str]) -> float:
        self.train()
        batch = self.prepare_indexing_batch(doc_texts, doc_ids)
        
        with torch.cuda.amp.autocast():
            outputs = self.forward(**batch)
            loss = outputs.loss
        
        return loss.item() if loss is not None else 0.0
    
    def retrieve_documents(self, queries: List[str], doc_ids: List[str]) -> float:
        self.train()
        batch = self.prepare_retrieval_batch(queries, doc_ids)
        
        with torch.cuda.amp.autocast():
            outputs = self.forward(**batch)
            loss = outputs.loss
        
        return loss.item() if loss is not None else 0.0
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        ground_truth_docids: List[str],
        valid_docids: Optional[List[str]] = None,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        self.eval()
        
        generated_docids = self.generate_docids(
            queries,
            num_return_sequences=max(k_values),
            valid_docids=valid_docids
        )
        
        metrics = {}
        for k in k_values:
            hits = 0
            for i, (generated, ground_truth) in enumerate(zip(generated_docids, ground_truth_docids)):
                top_k = generated[:k]
                if ground_truth in top_k:
                    hits += 1
            
            metrics[f'hits@{k}'] = hits / len(queries)
        
        return metrics
    
    def save_model(self, save_path: str):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def load_model(self, load_path: str):
        self.model = T5ForConditionalGeneration.from_pretrained(load_path)
        self.tokenizer = T5Tokenizer.from_pretrained(load_path)
        self.docid_tokenizer = DocIDTokenizer(self.tokenizer, self.docid_format)
    
    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
    
    def set_valid_docids(self, valid_docids: List[str]):
        if self.trie_constraint is not None:
            self.trie_constraint.build_trie(valid_docids)