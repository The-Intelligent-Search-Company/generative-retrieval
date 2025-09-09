from typing import List, Dict, Set, Optional, Tuple, Union
import torch
from transformers import PreTrainedTokenizer
from transformers.generation.utils import DisjunctiveConstraint
from transformers.generation.beam_constraints import Constraint


class TrieNode:
    def __init__(self):
        self.children: Dict[int, 'TrieNode'] = {}
        self.is_end: bool = False
        self.token_id: Optional[int] = None
    
    def add_child(self, token_id: int) -> 'TrieNode':
        if token_id not in self.children:
            self.children[token_id] = TrieNode()
            self.children[token_id].token_id = token_id
        return self.children[token_id]
    
    def get_child(self, token_id: int) -> Optional['TrieNode']:
        return self.children.get(token_id)
    
    def get_valid_tokens(self) -> List[int]:
        return list(self.children.keys())
    
    def mark_end(self):
        self.is_end = True


class DocIDTrie:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.root = TrieNode()
        self.docid_count = 0
    
    def add_docid(self, docid: str):
        token_ids = self.tokenizer.encode(docid, add_special_tokens=False)
        current = self.root
        
        for token_id in token_ids:
            current = current.add_child(token_id)
        
        current.mark_end()
        self.docid_count += 1
    
    def add_docids(self, docids: List[str]):
        for docid in docids:
            self.add_docid(docid)
    
    def is_valid_prefix(self, token_ids: List[int]) -> bool:
        current = self.root
        for token_id in token_ids:
            current = current.get_child(token_id)
            if current is None:
                return False
        return True
    
    def is_complete_docid(self, token_ids: List[int]) -> bool:
        current = self.root
        for token_id in token_ids:
            current = current.get_child(token_id)
            if current is None:
                return False
        return current.is_end
    
    def get_valid_next_tokens(self, token_ids: List[int]) -> List[int]:
        current = self.root
        for token_id in token_ids:
            current = current.get_child(token_id)
            if current is None:
                return []
        return current.get_valid_tokens()
    
    def get_all_completions(self, prefix_token_ids: List[int]) -> List[List[int]]:
        current = self.root
        for token_id in prefix_token_ids:
            current = current.get_child(token_id)
            if current is None:
                return []
        
        completions = []
        self._collect_completions(current, prefix_token_ids.copy(), completions)
        return completions
    
    def _collect_completions(self, node: TrieNode, current_path: List[int], completions: List[List[int]]):
        if node.is_end:
            completions.append(current_path.copy())
        
        for token_id, child in node.children.items():
            current_path.append(token_id)
            self._collect_completions(child, current_path, completions)
            current_path.pop()
    
    def clear(self):
        self.root = TrieNode()
        self.docid_count = 0


class TrieConstraint:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 20):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.trie = DocIDTrie(tokenizer)
        self._constraint = None
    
    def build_trie(self, valid_docids: List[str]):
        self.trie.clear()
        self.trie.add_docids(valid_docids)
        self._constraint = DocIDTrieConstraint(self.trie, self.max_length)
    
    def get_constraint(self):
        if self._constraint is None:
            raise RuntimeError("Trie not built. Call build_trie() first.")
        return self._constraint
    
    def is_valid_docid(self, docid: str) -> bool:
        token_ids = self.tokenizer.encode(docid, add_special_tokens=False)
        return self.trie.is_complete_docid(token_ids)
    
    def get_valid_continuations(self, partial_docid: str) -> List[str]:
        token_ids = self.tokenizer.encode(partial_docid, add_special_tokens=False)
        next_tokens = self.trie.get_valid_next_tokens(token_ids)
        
        continuations = []
        for token_id in next_tokens:
            token_str = self.tokenizer.decode([token_id])
            continuations.append(token_str)
        
        return continuations


class DocIDTrieConstraint(Constraint):
    def __init__(self, trie: DocIDTrie, max_length: int = 20):
        self.trie = trie
        self.max_length = max_length
        self.seqlen = max_length  # Required by HuggingFace Constraint base class
        self.current_sequence = []
        self.completed = False
        
    def advance(self) -> Union[int, List[int], None]:
        """Return tokens that advance the constraint towards being fulfilled."""
        if self.completed:
            return None
        
        valid_tokens = self.trie.get_valid_next_tokens(self.current_sequence)
        if not valid_tokens:
            return None
        
        # Return the first valid token as default advancement
        return valid_tokens[0] if len(valid_tokens) == 1 else valid_tokens
    
    def does_advance(self, token_id: int) -> bool:
        """Check if the given token advances the constraint."""
        if self.completed:
            return False
        
        valid_tokens = self.trie.get_valid_next_tokens(self.current_sequence)
        return token_id in valid_tokens
    
    def update(self, token_id: int) -> Tuple[bool, bool, bool]:
        """Update constraint state with new token.
        
        Returns:
            stepped: Whether constraint is closer to being fulfilled
            completed: Whether constraint is fully satisfied  
            reset: Whether progress was interrupted (need to start over)
        """
        if self.completed:
            return False, True, False
        
        valid_tokens = self.trie.get_valid_next_tokens(self.current_sequence)
        
        if token_id in valid_tokens:
            # Valid step forward
            new_sequence = self.current_sequence + [token_id]
            self.current_sequence = new_sequence
            
            # Check if this completes a valid DocID
            is_complete = self.trie.is_complete_docid(new_sequence)
            if is_complete:
                self.completed = True
                return True, True, False
            
            # Check if we can continue (valid prefix)
            is_valid_prefix = self.trie.is_valid_prefix(new_sequence)
            if is_valid_prefix:
                return True, False, False
            else:
                # Dead end, need to reset
                self.reset()
                return False, False, True
        else:
            # Invalid token, reset
            self.reset()
            return False, False, True
    
    def reset(self):
        """Reset constraint state."""
        self.current_sequence = []
        self.completed = False
    
    def remaining(self) -> int:
        """Return number of steps remaining to complete constraint."""
        if self.completed:
            return 0
        
        # Estimate remaining steps (this is approximate)
        current_length = len(self.current_sequence)
        if current_length == 0:
            # At least one token needed
            return 1
        
        # Check if current sequence can lead to completion
        valid_tokens = self.trie.get_valid_next_tokens(self.current_sequence)
        if not valid_tokens:
            return float('inf')  # Cannot be completed
        
        # Estimate based on minimum remaining tokens needed
        # This is a heuristic - exact calculation would require tree traversal
        return max(1, self.max_length - current_length)
    
    def copy(self, stateful=True):
        """Create a copy of this constraint."""
        new_constraint = DocIDTrieConstraint(self.trie, self.max_length)
        if stateful:
            new_constraint.current_sequence = self.current_sequence.copy()
            new_constraint.completed = self.completed
        else:
            # Non-stateful copy - start fresh
            new_constraint.current_sequence = []
            new_constraint.completed = False
        return new_constraint
    
    def __call__(self, batch_id: int, input_ids: torch.Tensor) -> torch.Tensor:
        """Legacy call method for backwards compatibility."""
        current_sequence = input_ids.tolist()
        
        if not current_sequence:
            valid_tokens = self.trie.get_valid_next_tokens([])
        else:
            valid_tokens = self.trie.get_valid_next_tokens(current_sequence)
        
        if not valid_tokens:
            return torch.tensor([], dtype=torch.long)
        
        return torch.tensor(valid_tokens, dtype=torch.long)


class BeamSearchConstraint:
    def __init__(self, tokenizer: PreTrainedTokenizer, valid_docids: Optional[List[str]] = None):
        self.tokenizer = tokenizer
        self.trie_constraint = TrieConstraint(tokenizer)
        
        if valid_docids:
            self.set_valid_docids(valid_docids)
    
    def set_valid_docids(self, valid_docids: List[str]):
        self.trie_constraint.build_trie(valid_docids)
    
    def filter_beam_candidates(
        self, 
        input_ids: torch.Tensor, 
        next_token_scores: torch.Tensor,
        beam_size: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        vocab_size = next_token_scores.shape[-1]
        
        filtered_scores = torch.full_like(next_token_scores, float('-inf'))
        
        for batch_idx in range(batch_size):
            current_sequence = input_ids[batch_idx].tolist()
            
            valid_tokens = self.trie_constraint.trie.get_valid_next_tokens(current_sequence)
            
            if valid_tokens:
                for token_id in valid_tokens:
                    if token_id < vocab_size:
                        filtered_scores[batch_idx, token_id] = next_token_scores[batch_idx, token_id]
            else:
                eos_token_id = self.tokenizer.eos_token_id
                if eos_token_id is not None and eos_token_id < vocab_size:
                    filtered_scores[batch_idx, eos_token_id] = next_token_scores[batch_idx, eos_token_id]
        
        return input_ids, filtered_scores
    
    def is_sequence_valid(self, token_ids: List[int]) -> bool:
        return self.trie_constraint.trie.is_complete_docid(token_ids)
    
    def get_completion_score(self, token_ids: List[int]) -> float:
        if self.is_sequence_valid(token_ids):
            return 1.0
        elif self.trie_constraint.trie.is_valid_prefix(token_ids):
            return 0.5
        else:
            return 0.0


class MultiLevelConstraint:
    def __init__(self, tokenizer: PreTrainedTokenizer, hierarchy_levels: List[int]):
        self.tokenizer = tokenizer
        self.hierarchy_levels = hierarchy_levels
        self.level_tries = [DocIDTrie(tokenizer) for _ in hierarchy_levels]
        self.level_separators = ["-", ".", "|"][:len(hierarchy_levels) - 1]
    
    def add_hierarchical_docids(self, hierarchical_docids: List[str]):
        for docid in hierarchical_docids:
            self._add_hierarchical_docid(docid)
    
    def _add_hierarchical_docid(self, docid: str):
        parts = docid.split("-")
        if len(parts) != len(self.hierarchy_levels):
            raise ValueError(f"Invalid hierarchical docid: {docid}")
        
        for level, part in enumerate(parts):
            self.level_tries[level].add_docid(part)
    
    def get_level_constraint(self, level: int) -> DocIDTrieConstraint:
        if 0 <= level < len(self.level_tries):
            return DocIDTrieConstraint(self.level_tries[level])
        raise ValueError(f"Invalid level: {level}")
    
    def validate_hierarchical_sequence(self, token_ids: List[int], current_level: int = 0) -> bool:
        if current_level >= len(self.hierarchy_levels):
            return True
        
        level_trie = self.level_tries[current_level]
        separator_tokens = self.tokenizer.encode("-", add_special_tokens=False) if current_level < len(self.hierarchy_levels) - 1 else []
        
        level_tokens = []
        i = 0
        
        while i < len(token_ids):
            if separator_tokens and token_ids[i:i+len(separator_tokens)] == separator_tokens:
                if level_trie.is_complete_docid(level_tokens):
                    remaining_tokens = token_ids[i+len(separator_tokens):]
                    return self.validate_hierarchical_sequence(remaining_tokens, current_level + 1)
                else:
                    return False
            else:
                level_tokens.append(token_ids[i])
                i += 1
        
        return level_trie.is_complete_docid(level_tokens) and current_level == len(self.hierarchy_levels) - 1