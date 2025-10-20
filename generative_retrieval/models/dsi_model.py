from typing import List, Dict, Any, Optional, Union
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from .constraints import TrieConstraint
from .tokenizers import DocIDTokenizer


class DSIModel(nn.Module):
    """Differentiable Search Index (DSI) model using T5 architecture.
    
    This model implements the DSI approach for neural information retrieval,
    where documents are mapped to identifiers that can be generated autoregressively.
    
    Args:
        model_name: Name of the T5 model to use (e.g., 't5-base', 't5-large')
        docid_format: Format for document IDs ('sequential', 'hierarchical', etc.)
        max_docid_length: Maximum length for generated document IDs
        use_constrained_generation: Whether to use trie-based constraints during generation
    """
    
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
            self.trie_constraint = TrieConstraint(self.tokenizer, max_docid_length)
    
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
            'labels': target_encodings['input_ids'],
            'task_type': 'indexing'
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
            'labels': target_encodings['input_ids'],
            'task_type': 'retrieval'
        }
    
    def generate_docids(
        self,
        queries: List[str],
        num_return_sequences: int = 10,
        valid_docids: Optional[List[str]] = None,
        do_sample: bool = True,
        temperature: float = 0.1
    ) -> List[List[str]]:
        inputs = [f"retrieve query: {query}" for query in queries]
        
        input_encodings = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Determine generation strategy based on constraints and sampling
        if self.use_constrained_generation and valid_docids is not None:
            # Constrained generation requires beam search
            generation_kwargs = {
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'max_length': self.max_docid_length,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'return_dict_in_generate': True,
                'output_scores': True,
                'num_beams': max(num_return_sequences, 2),
                'num_return_sequences': num_return_sequences,
                'early_stopping': True,
                'do_sample': False  # Constrained generation doesn't work well with sampling
            }
        else:
            # Regular generation (sampling or beam search)
            generation_kwargs = {
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'max_length': self.max_docid_length,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'return_dict_in_generate': True,
                'output_scores': True,
                'num_return_sequences': num_return_sequences
            }
            
            if do_sample:
                generation_kwargs.update({
                    'do_sample': True,
                    'temperature': temperature,
                    'top_p': 0.9,
                    'repetition_penalty': 1.1
                })
            elif num_return_sequences > 1:
                generation_kwargs.update({
                    'num_beams': max(num_return_sequences, 2),
                    'early_stopping': True,
                    'do_sample': False
                })
            else:
                # Greedy decoding for single sequence
                generation_kwargs.update({
                    'do_sample': False,
                    'temperature': 1.0
                })
        
        # Apply constraints if using constrained generation
        if self.use_constrained_generation and valid_docids is not None:
            self.trie_constraint.build_trie(valid_docids)
            generation_kwargs['constraints'] = [self.trie_constraint.get_constraint()]
            # Ensure we use beam search for constrained generation
            if 'do_sample' in generation_kwargs:
                generation_kwargs['do_sample'] = False
            if 'num_beams' not in generation_kwargs:
                generation_kwargs['num_beams'] = max(num_return_sequences, 2)
        
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
                
                # Clean up DocID formatting
                if self.docid_tokenizer:
                    docid = self.docid_tokenizer._postprocess_docid(docid)
                else:
                    # Basic cleanup for sequential DocIDs
                    docid = ''.join(filter(str.isdigit, docid))  # Keep only digits
                    if len(docid) > 0 and len(docid) < 8:
                        docid = docid.zfill(8)  # Pad to 8 digits
                    elif len(docid) > 8:
                        docid = docid[:8]  # Truncate to 8 digits
                
                # Validate DocID
                if docid and len(docid) == 8 and docid.isdigit():
                    # Only add valid DocIDs if using constraints
                    if (not self.use_constrained_generation or 
                        valid_docids is None or 
                        docid in valid_docids):
                        batch_results.append(docid)
                    elif valid_docids:  # Find closest valid DocID
                        try:
                            target_num = int(docid)
                            closest_docid = min(valid_docids, 
                                               key=lambda x: abs(int(x) - target_num) if x.isdigit() else float('inf'))
                            batch_results.append(closest_docid)
                        except (ValueError, TypeError):
                            # Fallback to a valid DocID
                            if valid_docids and valid_docids[0] not in batch_results:
                                batch_results.append(valid_docids[0])
            
            # Ensure we have enough results by filling with valid DocIDs
            if len(batch_results) < num_return_sequences and valid_docids:
                remaining_valid = [docid for docid in valid_docids if docid not in batch_results]
                for docid in remaining_valid[:num_return_sequences - len(batch_results)]:
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
    
    def compute_multitask_loss(
        self,
        indexing_batch: Optional[Dict[str, torch.Tensor]] = None,
        retrieval_batch: Optional[Dict[str, torch.Tensor]] = None,
        indexing_weight: float = 1.0,
        retrieval_weight: float = 1.0
    ) -> torch.Tensor:
        """Compute weighted multi-task loss for indexing and retrieval."""
        total_loss = 0.0
        loss_components = {}
        
        if indexing_batch is not None:
            indexing_outputs = self.forward(
                input_ids=indexing_batch['input_ids'],
                attention_mask=indexing_batch['attention_mask'],
                labels=indexing_batch['labels']
            )
            indexing_loss = indexing_outputs.loss * indexing_weight
            total_loss += indexing_loss
            loss_components['indexing'] = indexing_loss
        
        if retrieval_batch is not None:
            retrieval_outputs = self.forward(
                input_ids=retrieval_batch['input_ids'],
                attention_mask=retrieval_batch['attention_mask'],
                labels=retrieval_batch['labels']
            )
            retrieval_loss = retrieval_outputs.loss * retrieval_weight
            total_loss += retrieval_loss
            loss_components['retrieval'] = retrieval_loss
        
        # Create output object with loss and components for multi-task training
        from dataclasses import dataclass
        from typing import Dict
        
        @dataclass
        class MultiTaskOutput:
            loss: torch.Tensor
            loss_components: Dict[str, torch.Tensor]
        
        return MultiTaskOutput(loss=total_loss, loss_components=loss_components)