import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict

from .model import DSIMultiTaskModel, DSIModel
from .trainer import DSIDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DSIEvaluator:
    def __init__(
        self,
        model: DSIModel,
        eval_dataset_path: str,
        docid_mapping_path: Optional[str] = None,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        
        logger.info(f"Loading evaluation dataset from {eval_dataset_path}")
        eval_dataset = load_from_disk(eval_dataset_path)
        self.eval_dataset = DSIDataset(eval_dataset, model.tokenizer)
        
        if docid_mapping_path and Path(docid_mapping_path).exists():
            self.docid_to_metadata = {}
            with open(docid_mapping_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    self.docid_to_metadata[item["docid"]] = {
                        "original_id": item["doc_id_original"],
                        "text": item["text"]
                    }
        else:
            self.docid_to_metadata = {}
    
    def compute_hit_at_k(
        self,
        predictions: List[List[str]],
        ground_truth: List[str],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        hits = {f"Hit@{k}": 0 for k in k_values}
        
        for pred_list, gt in zip(predictions, ground_truth):
            for k in k_values:
                top_k_preds = pred_list[:k]
                if gt in top_k_preds:
                    hits[f"Hit@{k}"] += 1
        
        total = len(ground_truth)
        return {metric: count / total for metric, count in hits.items()}
    
    def compute_mrr(
        self,
        predictions: List[List[str]],
        ground_truth: List[str]
    ) -> float:
        reciprocal_ranks = []
        
        for pred_list, gt in zip(predictions, ground_truth):
            try:
                rank = pred_list.index(gt) + 1
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks)
    
    def evaluate(
        self,
        num_beams: int = 10,
        num_return_sequences: int = 10,
        k_values: List[int] = [1, 5, 10],
        task_filter: Optional[str] = None
    ) -> Dict[str, float]:
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        all_predictions = []
        all_ground_truth = []
        task_specific_metrics = defaultdict(lambda: {"predictions": [], "ground_truth": []})
        
        logger.info("Starting evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]
                task_types = batch["task_type"]
                
                ground_truth_docids = []
                for label in labels:
                    label_copy = label.clone()
                    label_copy[label_copy == -100] = self.model.tokenizer.pad_token_id
                    docid = self.model.tokenizer.decode(label_copy, skip_special_tokens=True)
                    ground_truth_docids.append(docid)
                
                if task_filter:
                    indices_to_eval = [i for i, t in enumerate(task_types) if t == task_filter]
                    if not indices_to_eval:
                        continue
                    input_ids = input_ids[indices_to_eval]
                    attention_mask = attention_mask[indices_to_eval]
                    ground_truth_docids = [ground_truth_docids[i] for i in indices_to_eval]
                    task_types = [task_types[i] for i in indices_to_eval]
                
                predicted_docids = self.model.generate_docids(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences
                )
                
                all_predictions.extend(predicted_docids)
                all_ground_truth.extend(ground_truth_docids)
                
                for i, task_type in enumerate(task_types):
                    task_specific_metrics[task_type]["predictions"].append(predicted_docids[i])
                    task_specific_metrics[task_type]["ground_truth"].append(ground_truth_docids[i])
        
        logger.info("Computing metrics...")
        
        overall_metrics = self.compute_hit_at_k(all_predictions, all_ground_truth, k_values)
        overall_metrics["MRR"] = self.compute_mrr(all_predictions, all_ground_truth)
        
        logger.info("\n=== Overall Metrics ===")
        for metric, value in overall_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        for task_type, data in task_specific_metrics.items():
            task_metrics = self.compute_hit_at_k(
                data["predictions"], 
                data["ground_truth"], 
                k_values
            )
            task_metrics["MRR"] = self.compute_mrr(data["predictions"], data["ground_truth"])
            
            logger.info(f"\n=== {task_type.capitalize()} Task Metrics ===")
            for metric, value in task_metrics.items():
                overall_metrics[f"{task_type}_{metric}"] = value
                logger.info(f"{metric}: {value:.4f}")
        
        return overall_metrics
    
    def analyze_failures(
        self,
        num_samples: int = 10,
        num_beams: int = 10,
        num_return_sequences: int = 10
    ) -> List[Dict]:
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=1,
            shuffle=False
        )
        
        failures = []
        
        logger.info("Analyzing failure cases...")
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Analyzing")):
                if len(failures) >= num_samples:
                    break
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]
                task_type = batch["task_type"][0]
                
                label_copy = labels[0].clone()
                label_copy[label_copy == -100] = self.model.tokenizer.pad_token_id
                ground_truth = self.model.tokenizer.decode(label_copy, skip_special_tokens=True)
                
                predictions = self.model.generate_docids(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences
                )[0]
                
                if ground_truth not in predictions[:1]:
                    input_text = self.model.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    
                    failure_case = {
                        "input": input_text,
                        "ground_truth": ground_truth,
                        "predictions": predictions[:5],
                        "task_type": task_type,
                        "hit_at_10": ground_truth in predictions
                    }
                    
                    if ground_truth in self.docid_to_metadata:
                        failure_case["ground_truth_text"] = self.docid_to_metadata[ground_truth]["text"][:200]
                    
                    failures.append(failure_case)
        
        logger.info(f"\nFound {len(failures)} failure cases")
        for i, failure in enumerate(failures[:5]):
            logger.info(f"\n--- Failure {i+1} ---")
            logger.info(f"Input: {failure['input'][:200]}")
            logger.info(f"Ground Truth: {failure['ground_truth']}")
            logger.info(f"Top Predictions: {failure['predictions']}")
            logger.info(f"Task: {failure['task_type']}")
        
        return failures
    
    def save_results(self, metrics: Dict, output_path: str):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved evaluation results to {output_file}")


def evaluate_checkpoint(
    checkpoint_path: str,
    eval_dataset_path: str,
    docid_mapping_path: Optional[str] = None,
    output_path: Optional[str] = None,
    num_beams: int = 10,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    logger.info(f"Loading model from {checkpoint_path}")
    
    model = DSIMultiTaskModel.from_pretrained(checkpoint_path)
    
    evaluator = DSIEvaluator(
        model=model.dsi_model,
        eval_dataset_path=eval_dataset_path,
        docid_mapping_path=docid_mapping_path
    )
    
    metrics = evaluator.evaluate(
        num_beams=num_beams,
        num_return_sequences=num_beams,
        k_values=k_values
    )
    
    failures = evaluator.analyze_failures(num_samples=20)
    
    if output_path:
        results = {
            "metrics": metrics,
            "num_failure_samples": len(failures),
            "checkpoint_path": checkpoint_path
        }
        evaluator.save_results(results, output_path)
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate DSI model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--docid_mapping", type=str, default=None, help="Path to docid mapping file")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output path for results")
    parser.add_argument("--num_beams", type=int, default=10, help="Number of beams for generation")
    
    args = parser.parse_args()
    
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        eval_dataset_path=args.eval_data,
        docid_mapping_path=args.docid_mapping,
        output_path=args.output,
        num_beams=args.num_beams
    )
