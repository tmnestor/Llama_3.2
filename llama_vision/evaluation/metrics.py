"""Performance metrics calculation for Llama-3.2-Vision package."""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils import setup_logging


class PerformanceMetrics:
    """Calculate performance metrics following InternVL pattern."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize performance metrics calculator.

        Args:
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)

    def calculate_field_accuracy(
        self, extracted: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate field-level accuracy metrics.

        Args:
            extracted: Extracted data from model
            ground_truth: Ground truth reference data

        Returns:
            Dictionary of accuracy metrics per field
        """
        field_accuracies = {}

        # Get all fields present in either dataset
        all_fields = set(extracted.keys()) | set(ground_truth.keys())

        for field in all_fields:
            extracted_value = extracted.get(field, "")
            ground_truth_value = ground_truth.get(field, "")

            # Calculate accuracy for this field
            accuracy = self._calculate_single_field_accuracy(
                extracted_value, ground_truth_value, field
            )
            field_accuracies[field] = accuracy

        return field_accuracies

    def _calculate_single_field_accuracy(
        self, extracted: Any, ground_truth: Any, field_name: str
    ) -> float:
        """Calculate accuracy for a single field.

        Args:
            extracted: Extracted value
            ground_truth: Ground truth value
            field_name: Name of the field

        Returns:
            Accuracy score (0.0 to 1.0)
        """
        # Handle empty/missing values
        if not extracted and not ground_truth:
            return 1.0  # Both empty = perfect match
        elif not extracted or not ground_truth:
            return 0.0  # One empty, one not = no match

        # Convert to strings for comparison
        extracted_str = str(extracted).strip().lower()
        ground_truth_str = str(ground_truth).strip().lower()

        # Exact match
        if extracted_str == ground_truth_str:
            return 1.0

        # Field-specific comparison logic
        if field_name.lower() in ["date", "invoice_date", "transaction_date"]:
            return self._compare_dates(extracted_str, ground_truth_str)
        elif field_name.lower() in ["total", "amount", "total_amount", "gst", "tax"]:
            return self._compare_amounts(extracted_str, ground_truth_str)
        elif field_name.lower() in ["store", "supplier_name", "business_name"]:
            return self._compare_business_names(extracted_str, ground_truth_str)
        elif field_name.lower() in ["abn", "supplier_abn"]:
            return self._compare_abns(extracted_str, ground_truth_str)
        elif isinstance(extracted, list) or isinstance(ground_truth, list):
            return self._compare_lists(extracted, ground_truth)
        else:
            # Generic string similarity
            return self._calculate_string_similarity(extracted_str, ground_truth_str)

    def _compare_dates(self, extracted: str, ground_truth: str) -> float:
        """Compare date fields with format flexibility."""
        # Remove common separators and normalize
        extracted_clean = re.sub(r"[/-]", "", extracted)
        ground_truth_clean = re.sub(r"[/-]", "", ground_truth)

        if extracted_clean == ground_truth_clean:
            return 1.0

        # Try to parse and compare dates
        try:
            # Common date formats
            formats = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d", "%d%m%Y"]

            extracted_date = None
            ground_truth_date = None

            for fmt in formats:
                try:
                    if not extracted_date:
                        extracted_date = datetime.strptime(extracted, fmt)
                except ValueError:
                    continue

                try:
                    if not ground_truth_date:
                        ground_truth_date = datetime.strptime(ground_truth, fmt)
                except ValueError:
                    continue

            if extracted_date and ground_truth_date:
                return 1.0 if extracted_date.date() == ground_truth_date.date() else 0.0

        except Exception:
            pass

        # Fallback to string similarity
        return self._calculate_string_similarity(extracted, ground_truth)

    def _compare_amounts(self, extracted: str, ground_truth: str) -> float:
        """Compare monetary amounts with format flexibility."""
        # Extract numeric values
        extracted_num = self._extract_numeric_value(extracted)
        ground_truth_num = self._extract_numeric_value(ground_truth)

        if extracted_num is None or ground_truth_num is None:
            return 0.0

        # Allow small rounding differences (within 1 cent)
        if abs(extracted_num - ground_truth_num) <= 0.01:
            return 1.0

        # Calculate relative error
        if ground_truth_num != 0:
            relative_error = abs(extracted_num - ground_truth_num) / ground_truth_num
            # Give partial credit for close values (within 5%)
            if relative_error <= 0.05:
                return 1.0 - relative_error

        return 0.0

    def _extract_numeric_value(self, value: str) -> Optional[float]:
        """Extract numeric value from string."""
        try:
            # Remove currency symbols and spaces
            cleaned = re.sub(r"[$AUD\s,]", "", value)
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def _compare_business_names(self, extracted: str, ground_truth: str) -> float:
        """Compare business names with normalization."""
        # Normalize business names
        extracted_norm = self._normalize_business_name(extracted)
        ground_truth_norm = self._normalize_business_name(ground_truth)

        if extracted_norm == ground_truth_norm:
            return 1.0

        # Check if one contains the other
        if extracted_norm in ground_truth_norm or ground_truth_norm in extracted_norm:
            return 0.8

        # String similarity with higher threshold for business names
        similarity = self._calculate_string_similarity(
            extracted_norm, ground_truth_norm
        )
        return similarity if similarity >= 0.7 else 0.0

    def _normalize_business_name(self, name: str) -> str:
        """Normalize business name for comparison."""
        # Remove common business suffixes and normalize
        name = name.upper()
        suffixes = [
            "PTY LTD",
            "PTY",
            "LTD",
            "LIMITED",
            "PROPRIETARY",
            "COMPANY",
            "CORP",
            "INC",
        ]

        for suffix in suffixes:
            name = name.replace(suffix, "").strip()

        # Remove extra spaces
        name = " ".join(name.split())

        return name

    def _compare_abns(self, extracted: str, ground_truth: str) -> float:
        """Compare Australian Business Numbers."""
        # Remove spaces and normalize
        extracted_clean = re.sub(r"\s", "", extracted)
        ground_truth_clean = re.sub(r"\s", "", ground_truth)

        return 1.0 if extracted_clean == ground_truth_clean else 0.0

    def _compare_lists(self, extracted: Any, ground_truth: Any) -> float:
        """Compare list fields (e.g., items, quantities)."""
        # Convert to lists if not already
        extracted_list = (
            extracted
            if isinstance(extracted, list)
            else [extracted]
            if extracted
            else []
        )
        ground_truth_list = (
            ground_truth
            if isinstance(ground_truth, list)
            else [ground_truth]
            if ground_truth
            else []
        )

        if not extracted_list and not ground_truth_list:
            return 1.0

        if not extracted_list or not ground_truth_list:
            return 0.0

        # Calculate Jaccard similarity
        extracted_set = set(str(item).lower().strip() for item in extracted_list)
        ground_truth_set = set(str(item).lower().strip() for item in ground_truth_list)

        intersection = len(extracted_set & ground_truth_set)
        union = len(extracted_set | ground_truth_set)

        return intersection / union if union > 0 else 0.0

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Levenshtein distance."""
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0

        # Calculate Levenshtein distance
        len1, len2 = len(str1), len(str2)

        # Create matrix
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        # Initialize first row and column
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        # Fill matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if str1[i - 1] == str2[j - 1]:
                    cost = 0
                else:
                    cost = 1

                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,  # Deletion
                    matrix[i][j - 1] + 1,  # Insertion
                    matrix[i - 1][j - 1] + cost,  # Substitution
                )

        # Calculate similarity (1 - normalized distance)
        max_len = max(len1, len2)
        similarity = 1 - (matrix[len1][len2] / max_len) if max_len > 0 else 1.0

        return max(0.0, similarity)

    def calculate_overall_metrics(
        self, field_accuracies: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate overall performance metrics.

        Args:
            field_accuracies: Dictionary of field-level accuracies

        Returns:
            Dictionary of overall metrics
        """
        if not field_accuracies:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 0.0}

        # Calculate basic metrics
        accuracies = list(field_accuracies.values())
        average_accuracy = sum(accuracies) / len(accuracies)

        # Calculate precision and recall
        # For document extraction, precision = accuracy of extracted fields
        # Recall = proportion of ground truth fields that were extracted
        precision = average_accuracy
        recall = average_accuracy  # Simplified for now

        # Calculate F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": average_accuracy,
            "field_count": len(field_accuracies),
        }

    def generate_evaluation_report(
        self, results: List[Dict[str, Any]], output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report.

        Args:
            results: List of evaluation results
            output_path: Optional path to save report

        Returns:
            Evaluation report dictionary
        """
        if not results:
            return {"error": "No results to evaluate"}

        # Calculate aggregate metrics
        all_field_accuracies = {}
        overall_metrics = []

        for result in results:
            if "field_accuracies" in result:
                field_acc = result["field_accuracies"]

                # Aggregate field accuracies
                for field, acc in field_acc.items():
                    if field not in all_field_accuracies:
                        all_field_accuracies[field] = []
                    all_field_accuracies[field].append(acc)

                # Calculate overall metrics for this result
                overall = self.calculate_overall_metrics(field_acc)
                overall_metrics.append(overall)

        # Calculate average field accuracies
        avg_field_accuracies = {
            field: sum(accs) / len(accs) for field, accs in all_field_accuracies.items()
        }

        # Calculate average overall metrics
        if overall_metrics:
            avg_overall = {
                metric: sum(result[metric] for result in overall_metrics)
                / len(overall_metrics)
                for metric in overall_metrics[0].keys()
            }
        else:
            avg_overall = {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0,
            }

        # Generate report
        report = {
            "summary": {
                "total_documents": len(results),
                "average_accuracy": avg_overall["accuracy"],
                "average_precision": avg_overall["precision"],
                "average_recall": avg_overall["recall"],
                "average_f1_score": avg_overall["f1_score"],
            },
            "field_performance": avg_field_accuracies,
            "detailed_results": results,
            "recommendations": self._generate_recommendations(
                avg_field_accuracies, avg_overall
            ),
        }

        # Save report if output path provided
        if output_path:
            import json
            from pathlib import Path

            report_path = Path(output_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with report_path.open("w") as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"Evaluation report saved to: {report_path}")

        return report

    def _generate_recommendations(
        self, field_accuracies: Dict[str, float], overall_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on performance metrics."""
        recommendations = []

        # Overall performance recommendations
        if overall_metrics["accuracy"] >= 0.9:
            recommendations.append("Excellent performance - ready for production use")
        elif overall_metrics["accuracy"] >= 0.8:
            recommendations.append("Good performance - consider minor improvements")
        elif overall_metrics["accuracy"] >= 0.7:
            recommendations.append(
                "Moderate performance - prompt optimization recommended"
            )
        else:
            recommendations.append(
                "Performance needs improvement - consider model fine-tuning"
            )

        # Field-specific recommendations
        low_performing_fields = [
            field for field, acc in field_accuracies.items() if acc < 0.7
        ]

        if low_performing_fields:
            recommendations.append(
                f"Improve extraction for: {', '.join(low_performing_fields)}"
            )

        # Australian tax compliance recommendations
        compliance_fields = ["supplier_abn", "gst_amount", "invoice_date"]
        missing_compliance = [
            field
            for field in compliance_fields
            if field not in field_accuracies or field_accuracies[field] < 0.8
        ]

        if missing_compliance:
            recommendations.append(
                f"Enhance Australian tax compliance for: {', '.join(missing_compliance)}"
            )

        return recommendations
