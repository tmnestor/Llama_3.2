"""
Evaluator for receipt information extraction using synthetic datasets.

This module provides comprehensive evaluation metrics for zero-shot receipt
information extraction systems.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class ReceiptExtractorEvaluator:
    """Evaluator for receipt information extraction using synthetic datasets."""
    
    def __init__(
        self,
        extractor,
        ground_truth_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize receipt extractor evaluator.
        
        Args:
            extractor: Initialized receipt extractor model
            ground_truth_path: Path to ground truth data file or directory
            output_dir: Path to save evaluation results
        """
        self.logger = logging.getLogger(__name__)
        self.extractor = extractor
        
        # Load ground truth data
        self.ground_truth = self._load_ground_truth(ground_truth_path)
        self.ground_truth_path = Path(ground_truth_path)
        self.logger.info(f"Loaded {len(self.ground_truth)} ground truth samples")
        
        # Set up output directory
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_ground_truth(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load ground truth data.
        
        Args:
            path: Path to ground truth file or directory
            
        Returns:
            List of ground truth samples
        """
        path = Path(path)
        
        if path.is_file():
            # Load single file
            if path.suffix.lower() == ".json":
                with path.open("r") as f:
                    data = json.load(f)
                    # Handle both single object and list of objects
                    return data if isinstance(data, list) else [data]
            elif path.suffix.lower() == ".csv":
                return pd.read_csv(path).to_dict(orient="records")
        elif path.is_dir():
            # Load all JSON files in directory
            data = []
            for json_file in path.glob("*.json"):
                with json_file.open("r") as f:
                    file_data = json.load(f)
                    if json_file.name == "metadata.json":
                        # Handle metadata files that might contain lists
                        if isinstance(file_data, list):
                            data.extend(file_data)
                        else:
                            data.append(file_data)
                    else:
                        # Handle individual JSON files
                        if isinstance(file_data, list):
                            data.extend(file_data)
                        else:
                            data.append(file_data)
            return data
        
        raise ValueError(f"Unsupported ground truth path: {path}")
    
    def evaluate(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate receipt extractor on ground truth data.
        
        Args:
            sample_size: Number of samples to evaluate (None for all)
            
        Returns:
            Evaluation metrics
        """
        # Select samples
        samples = self.ground_truth
        if sample_size and sample_size < len(samples):
            samples = np.random.choice(samples, size=sample_size, replace=False).tolist()
        
        self.logger.info(f"Evaluating on {len(samples)} samples")
        
        # Track metrics
        results = []
        field_metrics = defaultdict(lambda: {"correct": 0, "total": 0, "error": 0})
        
        # Process each sample
        for i, sample in enumerate(samples):
            self.logger.info(f"Processing sample {i+1}/{len(samples)}")
            
            # Get image path
            image_path = sample.get("image_path")
            if not image_path:
                # Try to construct image path from metadata
                filename = sample.get("filename")
                if filename:
                    # Look for images directory relative to ground truth
                    image_dir = self.ground_truth_path.parent / "images"
                    if not image_dir.exists():
                        # Try other common directory structures
                        image_dir = self.ground_truth_path.parent / "data" / "images"
                        if not image_dir.exists():
                            image_dir = self.ground_truth_path.parent
                    image_path = image_dir / filename
            
            if not image_path or not Path(image_path).exists():
                self.logger.warning(f"Image does not exist: {image_path}")
                continue
            
            # Extract all fields
            try:
                extraction = self.extractor.extract_all_fields(image_path)
                
                # Compare with ground truth for each field
                result = {
                    "image_path": str(image_path),
                    "metrics": {}
                }
                
                # Evaluate each field
                self._evaluate_field(result, sample, extraction, "store_name", field_metrics)
                self._evaluate_field(result, sample, extraction, "date", field_metrics)
                self._evaluate_field(result, sample, extraction, "total_amount", field_metrics)
                self._evaluate_field(result, sample, extraction, "receipt_id", field_metrics)
                self._evaluate_field(result, sample, extraction, "payment_method", field_metrics)
                
                # Items evaluation (if available)
                if "items" in sample and sample["items"] and "items" in extraction and extraction["items"]:
                    result["metrics"]["items"] = self._evaluate_items(sample["items"], extraction["items"])
                    
                    # Update field metrics for items
                    if result["metrics"]["items"]["match_rate"] >= 0.8:
                        field_metrics["items"]["correct"] += 1
                    elif result["metrics"]["items"]["match_rate"] >= 0.5:
                        field_metrics["items"]["partial"] = field_metrics["items"].get("partial", 0) + 1
                    
                    field_metrics["items"]["total"] += 1
                
                # Save full extraction for analysis
                result["extraction"] = extraction
                result["ground_truth"] = sample
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process sample {image_path}: {e}")
                field_metrics["overall"]["error"] += 1
                
        # Calculate overall metrics
        metrics = self._calculate_overall_metrics(field_metrics)
        
        # Save detailed results
        detailed_results_file = self.output_dir / "detailed_results.json"
        with detailed_results_file.open("w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary metrics
        metrics_summary_file = self.output_dir / "metrics_summary.json"
        with metrics_summary_file.open("w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _evaluate_field(
        self, 
        result: Dict[str, Any], 
        ground_truth: Dict[str, Any], 
        extraction: Dict[str, Any], 
        field: str,
        field_metrics: Dict[str, Dict[str, int]]
    ) -> None:
        """Evaluate a specific extraction field.
        
        Args:
            result: Result dictionary to update
            ground_truth: Ground truth data
            extraction: Extracted data
            field: Field name to evaluate
            field_metrics: Metrics dictionary to update
        """
        # Skip if field is not in ground truth
        if field not in ground_truth or not ground_truth[field]:
            return
        
        # Get values
        gt_value = str(ground_truth[field]).strip().lower()
        extracted_value = str(extraction.get(field, "")).strip().lower()
        
        # Check for exact match
        exact_match = gt_value == extracted_value
        
        # Check for partial match (for longer text fields)
        partial_match = False
        if len(gt_value) > 5:
            # Calculate string similarity
            partial_match = (
                extracted_value in gt_value or
                gt_value in extracted_value or
                self._string_similarity(gt_value, extracted_value) > 0.8
            )
        
        # Update metrics
        result["metrics"][field] = {
            "ground_truth": ground_truth[field],
            "extracted": extraction.get(field),
            "exact_match": exact_match,
            "partial_match": partial_match if not exact_match else False
        }
        
        # Update field metrics
        field_metrics[field]["total"] += 1
        if exact_match:
            field_metrics[field]["correct"] += 1
            field_metrics["overall"]["correct"] += 1
        elif partial_match:
            field_metrics[field]["partial"] = field_metrics[field].get("partial", 0) + 1
            field_metrics["overall"]["partial"] = field_metrics["overall"].get("partial", 0) + 1
        
        field_metrics["overall"]["total"] += 1
    
    def _evaluate_items(self, ground_truth_items: List[Dict], extracted_items: List[Dict]) -> Dict[str, Any]:
        """Evaluate extracted items against ground truth.
        
        Args:
            ground_truth_items: List of ground truth items
            extracted_items: List of extracted items
            
        Returns:
            Item evaluation metrics
        """
        # Count matching items
        matched_items = 0
        partial_matches = 0
        
        # Track exact item matches
        item_matches = []
        
        # Compare each ground truth item with extracted items
        for gt_item in ground_truth_items:
            gt_name = str(gt_item.get("item_name", "")).lower()
            gt_price = str(gt_item.get("price", "")).lower()
            
            best_match = None
            best_score = 0
            
            # Find best matching extracted item
            for ext_item in extracted_items:
                ext_name = str(ext_item.get("item_name", "")).lower()
                ext_price = str(ext_item.get("price", "")).lower()
                
                # Calculate match score
                name_sim = self._string_similarity(gt_name, ext_name)
                
                # Price exact match adds weight
                price_match = gt_price in ext_price or ext_price in gt_price
                
                # Combined score
                score = name_sim + (0.5 if price_match else 0)
                
                # Update best match
                if score > best_score:
                    best_score = score
                    best_match = {
                        "item": ext_item,
                        "score": score,
                        "name_sim": name_sim,
                        "price_match": price_match
                    }
            
            # Record match quality
            if best_match:
                item_matches.append(best_match)
                
                if best_match["score"] > 1.3:  # High confidence match
                    matched_items += 1
                elif best_match["score"] > 0.8:  # Partial match
                    partial_matches += 1
        
        # Calculate metrics
        total_gt_items = len(ground_truth_items)
        total_extracted_items = len(extracted_items)
        
        precision = matched_items / total_extracted_items if total_extracted_items > 0 else 0
        recall = matched_items / total_gt_items if total_gt_items > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Include partial matches in match rate
        match_rate = (matched_items + 0.5 * partial_matches) / total_gt_items if total_gt_items > 0 else 0
        
        return {
            "matched_items": matched_items,
            "partial_matches": partial_matches,
            "total_gt_items": total_gt_items,
            "total_extracted_items": total_extracted_items,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "match_rate": match_rate
        }
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity (simple ratio-based approach).
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0-1)
        """
        # Simple implementation - in production, use difflib or another string similarity library
        if not str1 or not str2:
            return 0
            
        # Check for substring
        if str1 in str2 or str2 in str1:
            shorter = min(len(str1), len(str2))
            longer = max(len(str1), len(str2))
            return shorter / longer
        
        # Count common characters
        common = sum(1 for c in str1 if c in str2)
        total = max(len(str1), len(str2))
        
        return common / total if total > 0 else 0
    
    def _calculate_overall_metrics(self, field_metrics: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """Calculate overall evaluation metrics.
        
        Args:
            field_metrics: Field-level metrics
            
        Returns:
            Overall metrics dictionary
        """
        metrics = {
            "fields": {},
            "overall": {}
        }
        
        # Calculate per-field metrics
        for field, stats in field_metrics.items():
            if field == "overall":
                continue
                
            total = stats["total"]
            if total == 0:
                continue
                
            correct = stats["correct"]
            partial = stats.get("partial", 0)
            
            accuracy = correct / total
            # Include partial matches at half weight
            match_rate = (correct + 0.5 * partial) / total
            
            metrics["fields"][field] = {
                "accuracy": accuracy,
                "match_rate": match_rate,
                "count": total
            }
        
        # Calculate overall metrics
        overall = field_metrics["overall"]
        total = overall["total"]
        if total > 0:
            metrics["overall"] = {
                "accuracy": overall["correct"] / total,
                "match_rate": (overall["correct"] + 0.5 * overall.get("partial", 0)) / total,
                "error_rate": overall.get("error", 0) / (total + overall.get("error", 0)),
                "total_fields_evaluated": total
            }
        
        return metrics