from typing import List, Dict, Any, Optional, Union
import re
from transformers import PreTrainedTokenizer


class DocIDTokenizer:
    def __init__(self, base_tokenizer: PreTrainedTokenizer, docid_format: str = "sequential"):
        self.base_tokenizer = base_tokenizer
        self.docid_format = docid_format
        self._setup_docid_patterns()
    
    def _setup_docid_patterns(self):
        if self.docid_format == "sequential":
            self.docid_pattern = re.compile(r'^\d{8}$')
        elif self.docid_format == "hierarchical":
            self.docid_pattern = re.compile(r'^\d+-\d+-\d+$')
        elif self.docid_format == "semantic":
            self.docid_pattern = re.compile(r'^\d+-\d+-\d+$')
        elif self.docid_format == "hash":
            self.docid_pattern = re.compile(r'^hash_[a-f0-9]{12}$')
        else:
            self.docid_pattern = None
    
    def encode_docid(self, docid: str) -> List[int]:
        if self.docid_format == "hierarchical" and "-" in docid:
            return self._encode_hierarchical_docid(docid)
        else:
            return self.base_tokenizer.encode(docid, add_special_tokens=False)
    
    def _encode_hierarchical_docid(self, docid: str) -> List[int]:
        parts = docid.split("-")
        token_ids = []
        
        for i, part in enumerate(parts):
            if i > 0:
                sep_tokens = self.base_tokenizer.encode("-", add_special_tokens=False)
                token_ids.extend(sep_tokens)
            
            part_tokens = self.base_tokenizer.encode(part, add_special_tokens=False)
            token_ids.extend(part_tokens)
        
        return token_ids
    
    def decode_docid(self, token_ids: List[int]) -> str:
        decoded = self.base_tokenizer.decode(token_ids, skip_special_tokens=True)
        return self._postprocess_docid(decoded)
    
    def _postprocess_docid(self, docid: str) -> str:
        docid = docid.strip()
        
        if self.docid_format == "hierarchical":
            docid = re.sub(r'\s*-\s*', '-', docid)
        elif self.docid_format == "sequential":
            docid = re.sub(r'\D', '', docid)
            if docid and len(docid) < 8:
                docid = docid.zfill(8)
        
        return docid
    
    def validate_docid(self, docid: str) -> bool:
        if self.docid_pattern is None:
            return True
        return bool(self.docid_pattern.match(docid))
    
    def batch_encode_docids(self, docids: List[str]) -> List[List[int]]:
        return [self.encode_docid(docid) for docid in docids]
    
    def batch_decode_docids(self, token_ids_list: List[List[int]]) -> List[str]:
        return [self.decode_docid(token_ids) for token_ids in token_ids_list]
    
    def get_docid_vocab_size(self) -> int:
        if self.docid_format == "sequential":
            return 10
        elif self.docid_format == "hierarchical":
            return 11
        else:
            return self.base_tokenizer.vocab_size
    
    def get_max_docid_length(self) -> int:
        if self.docid_format == "sequential":
            return 8
        elif self.docid_format == "hierarchical":
            return 16
        elif self.docid_format == "hash":
            return 20
        else:
            return 32
    
    def create_docid_mask(self, docids: List[str]) -> List[List[bool]]:
        masks = []
        for docid in docids:
            token_ids = self.encode_docid(docid)
            mask = [True] * len(token_ids)
            masks.append(mask)
        return masks
    
    def truncate_docid(self, docid: str, max_length: int) -> str:
        token_ids = self.encode_docid(docid)
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        return self.decode_docid(token_ids)
    
    def pad_docids(self, docids: List[str], max_length: Optional[int] = None) -> List[str]:
        if max_length is None:
            max_length = max(len(self.encode_docid(docid)) for docid in docids)
        
        padded_docids = []
        for docid in docids:
            token_ids = self.encode_docid(docid)
            
            if len(token_ids) < max_length:
                pad_token_id = self.base_tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.base_tokenizer.eos_token_id
                
                token_ids.extend([pad_token_id] * (max_length - len(token_ids)))
            elif len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            padded_docids.append(self.decode_docid(token_ids))
        
        return padded_docids


class HierarchicalDocIDTokenizer(DocIDTokenizer):
    def __init__(self, base_tokenizer: PreTrainedTokenizer, hierarchy_levels: List[int] = [100, 10, 5]):
        super().__init__(base_tokenizer, "hierarchical")
        self.hierarchy_levels = hierarchy_levels
        self.level_separators = ["-", ".", "|"][:len(hierarchy_levels) - 1]
    
    def encode_hierarchical_path(self, level_indices: List[int]) -> str:
        if len(level_indices) != len(self.hierarchy_levels):
            raise ValueError(f"Expected {len(self.hierarchy_levels)} levels, got {len(level_indices)}")
        
        parts = []
        for i, (index, max_val) in enumerate(zip(level_indices, self.hierarchy_levels)):
            if index >= max_val:
                raise ValueError(f"Index {index} exceeds maximum {max_val} for level {i}")
            parts.append(str(index))
        
        return "-".join(parts)
    
    def decode_hierarchical_path(self, docid: str) -> List[int]:
        parts = docid.split("-")
        if len(parts) != len(self.hierarchy_levels):
            raise ValueError(f"Invalid hierarchical docid format: {docid}")
        
        try:
            return [int(part) for part in parts]
        except ValueError:
            raise ValueError(f"Invalid hierarchical docid format: {docid}")
    
    def get_level_vocab_size(self, level: int) -> int:
        if 0 <= level < len(self.hierarchy_levels):
            return self.hierarchy_levels[level]
        raise ValueError(f"Invalid level {level}")
    
    def validate_hierarchical_docid(self, docid: str) -> bool:
        try:
            path = self.decode_hierarchical_path(docid)
            return all(0 <= idx < max_val for idx, max_val in zip(path, self.hierarchy_levels))
        except ValueError:
            return False


class SemanticDocIDTokenizer(HierarchicalDocIDTokenizer):
    def __init__(
        self,
        base_tokenizer: PreTrainedTokenizer,
        cluster_sizes: List[int] = [100, 10, 5],
        use_semantic_tokens: bool = True
    ):
        super().__init__(base_tokenizer, cluster_sizes)
        self.use_semantic_tokens = use_semantic_tokens
        self.semantic_token_map = {}
        
        if use_semantic_tokens:
            self._initialize_semantic_tokens()
    
    def _initialize_semantic_tokens(self):
        for level in range(len(self.hierarchy_levels)):
            for cluster_id in range(self.hierarchy_levels[level]):
                semantic_token = f"<cluster_{level}_{cluster_id}>"
                if semantic_token not in self.base_tokenizer.get_vocab():
                    self.semantic_token_map[f"{level}_{cluster_id}"] = semantic_token
    
    def encode_semantic_docid(self, level_indices: List[int]) -> List[int]:
        if not self.use_semantic_tokens:
            return super().encode_docid(self.encode_hierarchical_path(level_indices))
        
        token_ids = []
        for level, cluster_id in enumerate(level_indices):
            semantic_key = f"{level}_{cluster_id}"
            if semantic_key in self.semantic_token_map:
                token = self.semantic_token_map[semantic_key]
                token_ids.extend(self.base_tokenizer.encode(token, add_special_tokens=False))
            else:
                token_ids.extend(self.base_tokenizer.encode(str(cluster_id), add_special_tokens=False))
            
            if level < len(level_indices) - 1:
                token_ids.extend(self.base_tokenizer.encode("-", add_special_tokens=False))
        
        return token_ids