#!/usr/bin/env python3
"""Prompt configuration loader for Llama-3.2-Vision system.

Following InternVL PoC best practices for YAML prompt management.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class PromptConfig:
    """Manages prompt configuration loading and selection."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize prompt configuration.
        
        Args:
            config_path: Path to prompts.yaml file. If None, uses environment
                        variable or searches for file in current directory.
        """
        self.prompts: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self._load_prompts(config_path)
    
    def _load_prompts(self, config_path: Optional[str] = None) -> None:
        """Load prompts from YAML configuration file."""
        # Determine config file path
        if config_path:
            prompts_path = Path(config_path)
        else:
            # Check environment variable first
            env_path = os.getenv('LLAMA_VISION_PROMPTS_PATH')
            if env_path:
                prompts_path = Path(env_path)
            else:
                # Search in current directory and common locations
                search_paths = [
                    Path('./prompts.yaml'),
                    Path('./config/prompts.yaml'), 
                    Path('../prompts.yaml'),
                    Path('./data/prompts.yaml'),
                ]
                
                prompts_path = None
                for path in search_paths:
                    if path.exists():
                        prompts_path = path
                        break
                
                if prompts_path is None:
                    raise FileNotFoundError(
                        "Could not find prompts.yaml file. Please specify path or set "
                        "LLAMA_VISION_PROMPTS_PATH environment variable."
                    )
        
        # Load YAML configuration
        try:
            with prompts_path.open('r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Extract prompts and metadata
            self.metadata = config.pop('prompt_metadata', {})
            self.prompts = config
            
            print(f"✅ Loaded {len(self.prompts)} prompts from {prompts_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load prompts from {prompts_path}: {e}") from e
    
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
    
    def get_prompt_for_document_type(self, document_type: str) -> str:
        """Get the recommended prompt for a specific document type.
        
        Args:
            document_type: Type of document (receipt, tax_invoice, etc.)
            
        Returns:
            The prompt text for that document type
        """
        type_mapping = self.metadata.get('document_type_mapping', {})
        prompt_name = type_mapping.get(document_type, 'key_value_receipt_prompt')
        
        return self.get_prompt(prompt_name)
    
    def get_recommended_prompts(self) -> List[str]:
        """Get list of recommended prompt names for production use."""
        return self.metadata.get('recommended_prompts', ['key_value_receipt_prompt'])
    
    def get_fallback_prompts(self) -> List[str]:
        """Get list of fallback prompts to try if primary extraction fails."""
        return self.metadata.get('fallback_chain', ['key_value_receipt_prompt'])
    
    def list_prompts(self) -> List[str]:
        """Get list of all available prompt names."""
        return list(self.prompts.keys())
    
    def get_testing_prompts(self) -> List[str]:
        """Get prompts suitable for testing and debugging."""
        return self.metadata.get('testing_prompts', ['vision_test_prompt'])


def load_prompt_config(config_path: Optional[str] = None) -> PromptConfig:
    """Load prompt configuration from YAML file.
    
    Args:
        config_path: Optional path to prompts.yaml file
        
    Returns:
        PromptConfig instance
    """
    return PromptConfig(config_path)


def get_prompt_with_fallback(
    prompt_config: PromptConfig, 
    primary_prompt: str,
    fallback_prompts: Optional[List[str]] = None
) -> str:
    """Get a prompt with fallback options.
    
    Args:
        prompt_config: PromptConfig instance
        primary_prompt: Primary prompt name to try
        fallback_prompts: List of fallback prompt names
        
    Returns:
        The first available prompt text
    """
    # Try primary prompt first
    try:
        return prompt_config.get_prompt(primary_prompt)
    except KeyError:
        pass
    
    # Try fallback prompts
    if fallback_prompts is None:
        fallback_prompts = prompt_config.get_fallback_prompts()
    
    for fallback in fallback_prompts:
        try:
            return prompt_config.get_prompt(fallback)
        except KeyError:
            continue
    
    # Last resort - return a basic prompt
    return "<|image|>Extract information from this receipt: date, store, total, tax."


# Example usage and testing
if __name__ == "__main__":
    # Test the prompt configuration
    try:
        config = load_prompt_config()
        
        print(f"Loaded prompts: {', '.join(config.list_prompts())}")
        print(f"Recommended prompts: {config.get_recommended_prompts()}")
        
        # Test getting specific prompts
        receipt_prompt = config.get_prompt_for_document_type('receipt')
        print(f"\nReceipt prompt preview: {receipt_prompt[:100]}...")
        
        # Test fallback mechanism
        safe_prompt = get_prompt_with_fallback(config, 'nonexistent_prompt')
        print(f"\nFallback prompt preview: {safe_prompt[:100]}...")
        
    except Exception as e:
        print(f"Error testing prompt config: {e}")