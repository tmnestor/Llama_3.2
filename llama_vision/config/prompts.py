"""Prompt management for Llama-3.2-Vision package."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .settings import get_project_root


class PromptManager:
    """Manage prompts following InternVL pattern."""

    def __init__(self, prompts_path: Optional[str] = None):
        """Initialize prompt manager.

        Args:
            prompts_path: Path to prompts.yaml file. If None, uses environment
                         variable or searches for file in current directory.
        """
        self.prompts: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self._load_prompts(prompts_path)

    def _load_prompts(self, prompts_path: Optional[str] = None) -> None:
        """Load prompts from YAML configuration file."""
        # Determine config file path
        if prompts_path:
            config_path = Path(prompts_path)
        else:
            # Check environment variable first
            env_path = os.getenv("LLAMA_VISION_PROMPTS_PATH")
            if env_path:
                config_path = Path(env_path)
            else:
                # Search in current directory and common locations
                project_root = get_project_root()
                search_paths = [
                    project_root / "prompts.yaml",
                    Path("./prompts.yaml"),
                    Path("./config/prompts.yaml"),
                    Path("../prompts.yaml"),
                    Path("./data/prompts.yaml"),
                ]

                config_path = None
                for path in search_paths:
                    if path.exists():
                        config_path = path
                        break

                if config_path is None:
                    raise FileNotFoundError(
                        "Could not find prompts.yaml file. Please specify path or set "
                        "LLAMA_VISION_PROMPTS_PATH environment variable."
                    )

        # Load YAML configuration
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Extract prompts and metadata
            self.metadata = config.pop("prompt_metadata", {})
            self.prompts = config

            print(f"✅ Loaded {len(self.prompts)} prompts from {config_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load prompts from {config_path}: {e}") from e

    def get_prompt(self, prompt_name: str) -> str:
        """Get a specific prompt by name.

        Args:
            prompt_name: Name of the prompt to retrieve

        Returns:
            The prompt text

        Raises:
            KeyError: If prompt name not found
        """
        if prompt_name not in self.prompts:
            available = list(self.prompts.keys())
            raise KeyError(
                f"Prompt '{prompt_name}' not found. Available prompts: {available}"
            )

        return self.prompts[prompt_name]

    def get_prompt_for_document_type(
        self, document_type: str, classification_response: str = ""
    ) -> str:
        """Get the recommended prompt for a specific document type.

        Args:
            document_type: Type of document (receipt, tax_invoice, etc.)
            classification_response: The model's classification response (for content analysis)

        Returns:
            The prompt text for that document type
        """
        # Smart content-aware prompt selection for Australian tax invoices
        if document_type == "tax_invoice" and classification_response:
            # Check if this tax invoice contains fuel indicators
            fuel_indicators = [
                "costco",
                "ulp",
                "unleaded",
                "diesel",
                "litre",
                " l ",
                "fuel",
                "petrol",
            ]
            response_lower = classification_response.lower()

            if any(indicator in response_lower for indicator in fuel_indicators):
                # This is a fuel tax invoice - use fuel-specific prompt
                return self.get_prompt("fuel_receipt_extraction_prompt")

        # Default mapping
        type_mapping = self.metadata.get("document_type_mapping", {})
        prompt_name = type_mapping.get(document_type, "key_value_receipt_prompt")

        return self.get_prompt(prompt_name)

    def get_recommended_prompts(self) -> List[str]:
        """Get list of recommended prompt names for production use."""
        return self.metadata.get("recommended_prompts", ["key_value_receipt_prompt"])

    def get_fallback_prompts(self) -> List[str]:
        """Get list of fallback prompts to try if primary extraction fails."""
        return self.metadata.get("fallback_chain", ["key_value_receipt_prompt"])

    def list_prompts(self) -> List[str]:
        """Get list of all available prompt names."""
        return list(self.prompts.keys())

    def get_testing_prompts(self) -> List[str]:
        """Get prompts suitable for testing and debugging."""
        return self.metadata.get("testing_prompts", ["vision_test_prompt"])

    def get_prompt_with_fallback(
        self, primary_prompt: str, fallback_prompts: Optional[List[str]] = None
    ) -> str:
        """Get a prompt with fallback options.

        Args:
            primary_prompt: Primary prompt name to try
            fallback_prompts: List of fallback prompt names

        Returns:
            The first available prompt text
        """
        # Try primary prompt first
        try:
            return self.get_prompt(primary_prompt)
        except KeyError:
            pass

        # Try fallback prompts
        if fallback_prompts is None:
            fallback_prompts = self.get_fallback_prompts()

        for fallback in fallback_prompts:
            try:
                return self.get_prompt(fallback)
            except KeyError:
                continue

        # Last resort - return a basic prompt
        return (
            "<|image|>Extract information from this receipt: date, store, total, tax."
        )
