from typing import Any, Dict, Callable, Union


def generate_sequential_ids(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    example['doc_id'] = f"{idx:08d}"
    return example


def add_doc_ids(dataset, method: Union[str, Callable] = "sequential"):
    if isinstance(method, str):
        if method == "sequential":
            return dataset.map(generate_sequential_ids, with_indices=True)
        else:
            raise ValueError(f"Unsupported doc_id generation method: '{method}'. Only 'sequential' is supported.")
    elif callable(method):
        # Custom function provided - use it directly
        return dataset.map(method, with_indices=True)
    else:
        raise TypeError(f"Method must be a string or callable function, got {type(method)}")