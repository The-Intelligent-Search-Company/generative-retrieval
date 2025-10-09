import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from typing import Optional, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConstrainedBeamSearchLogitsProcessor:
    def __init__(self, tokenizer: T5Tokenizer, valid_docids: List[str], max_docid_length: int = 20):
        self.tokenizer = tokenizer
        self.valid_docids = set(valid_docids)
        self.max_docid_length = max_docid_length
        
        self.docid_trie = self._build_trie(valid_docids)
        
    def _build_trie(self, docids: List[str]) -> Dict:
        trie = {}
        for docid in docids:
            tokens = self.tokenizer.encode(docid, add_special_tokens=False)
            current = trie
            for token in tokens:
                if token not in current:
                    current[token] = {}
                current = current[token]
            current['_end_'] = True
        return trie
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            generated_tokens = input_ids[batch_idx, 1:].tolist()
            
            current_trie = self.docid_trie
            for token in generated_tokens:
                if token in current_trie:
                    current_trie = current_trie[token]
                else:
                    scores[batch_idx, :] = float('-inf')
                    scores[batch_idx, self.tokenizer.eos_token_id] = 0
                    break
            else:
                valid_next_tokens = list(current_trie.keys())
                
                if '_end_' in valid_next_tokens:
                    valid_next_tokens.remove('_end_')
                    valid_next_tokens.append(self.tokenizer.eos_token_id)
                
                if len(valid_next_tokens) > 0:
                    mask = torch.ones_like(scores[batch_idx], dtype=torch.bool)
                    mask[valid_next_tokens] = False
                    scores[batch_idx, mask] = float('-inf')
                else:
                    scores[batch_idx, :] = float('-inf')
                    scores[batch_idx, self.tokenizer.eos_token_id] = 0
                    
        return scores


class DSIModel(nn.Module):
    def __init__(
        self,
        model_name: str = "t5-base",
        valid_docids: Optional[List[str]] = None,
        use_constrained_generation: bool = True,
        max_docid_length: int = 20
    ):
        super().__init__()
        
        logger.info(f"Initializing DSI model with backbone: {model_name}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        self.valid_docids = valid_docids or []
        self.use_constrained_generation = use_constrained_generation and len(self.valid_docids) > 0
        self.max_docid_length = max_docid_length
        
        if self.use_constrained_generation:
            self.logits_processor = ConstrainedBeamSearchLogitsProcessor(
                self.tokenizer, self.valid_docids, max_docid_length
            )
        else:
            self.logits_processor = None
            
        self.config = self.model.config
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits
        }
    
    def generate_docids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_beams: int = 10,
        num_return_sequences: int = 10,
        **kwargs
    ) -> List[List[str]]:
        if self.use_constrained_generation:
            from transformers import LogitsProcessorList
            logits_processor = LogitsProcessorList([self.logits_processor])
        else:
            logits_processor = None
            
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_docid_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            logits_processor=logits_processor,
            **kwargs
        )
        
        batch_size = input_ids.shape[0]
        generated_docids = []
        
        for i in range(batch_size):
            start_idx = i * num_return_sequences
            end_idx = start_idx + num_return_sequences
            batch_docids = [
                self.tokenizer.decode(generated_ids[j], skip_special_tokens=True)
                for j in range(start_idx, end_idx)
            ]
            generated_docids.append(batch_docids)
            
        return generated_docids
    
    def update_valid_docids(self, new_docids: List[str]):
        self.valid_docids = new_docids
        if self.use_constrained_generation and len(self.valid_docids) > 0:
            self.logits_processor = ConstrainedBeamSearchLogitsProcessor(
                self.tokenizer, self.valid_docids, self.max_docid_length
            )
    
    def save_pretrained(self, save_directory: str):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        import json
        metadata = {
            "valid_docids": self.valid_docids,
            "use_constrained_generation": self.use_constrained_generation,
            "max_docid_length": self.max_docid_length
        }
        with open(f"{save_directory}/dsi_metadata.json", "w") as f:
            json.dump(metadata, f)
            
    @classmethod
    def from_pretrained(cls, load_directory: str):
        import json
        with open(f"{load_directory}/dsi_metadata.json", "r") as f:
            metadata = json.load(f)
            
        model = cls(
            model_name=load_directory,
            valid_docids=metadata["valid_docids"],
            use_constrained_generation=metadata["use_constrained_generation"],
            max_docid_length=metadata["max_docid_length"]
        )
        
        return model


class DSIMultiTaskModel(nn.Module):
    def __init__(
        self,
        model_name: str = "t5-base",
        valid_docids: Optional[List[str]] = None,
        use_constrained_generation: bool = True,
        indexing_weight: float = 1.0,
        retrieval_weight: float = 1.0
    ):
        super().__init__()
        
        self.dsi_model = DSIModel(
            model_name=model_name,
            valid_docids=valid_docids,
            use_constrained_generation=use_constrained_generation
        )
        
        self.indexing_weight = indexing_weight
        self.retrieval_weight = retrieval_weight
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        task_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        outputs = self.dsi_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        if task_types is not None and len(task_types) == input_ids.shape[0]:
            loss = outputs["loss"]
            
            weighted_loss = 0.0
            for i, task_type in enumerate(task_types):
                weight = self.indexing_weight if task_type == "indexing" else self.retrieval_weight
                
                task_loss = loss if loss.dim() == 0 else loss[i]
                weighted_loss += weight * task_loss
                
            outputs["loss"] = weighted_loss / len(task_types)
        
        return outputs
    
    def generate_docids(self, *args, **kwargs):
        return self.dsi_model.generate_docids(*args, **kwargs)
    
    def save_pretrained(self, save_directory: str):
        self.dsi_model.save_pretrained(save_directory)
        
        import json
        with open(f"{save_directory}/multitask_config.json", "w") as f:
            json.dump({
                "indexing_weight": self.indexing_weight,
                "retrieval_weight": self.retrieval_weight
            }, f)
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        import json
        with open(f"{load_directory}/multitask_config.json", "r") as f:
            config = json.load(f)
            
        dsi_model = DSIModel.from_pretrained(load_directory)
        
        model = cls(
            model_name=load_directory,
            valid_docids=dsi_model.valid_docids,
            use_constrained_generation=dsi_model.use_constrained_generation,
            indexing_weight=config["indexing_weight"],
            retrieval_weight=config["retrieval_weight"]
        )
        model.dsi_model = dsi_model
        
        return model
