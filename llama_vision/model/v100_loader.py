"""V100-optimized model loading for Llama-3.2-Vision with 16GB memory constraint."""

import gc
from typing import Any, Tuple

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
)

from ..config import LlamaConfig
from ..utils import setup_logging


class V100ModelLoader:
    """Model loader optimized for V100 16GB deployment."""

    def __init__(self, config: LlamaConfig):
        """Initialize V100-optimized loader."""
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.target_memory_gb = 16.0  # V100 memory constraint

    def get_quantization_options(self) -> dict[str, dict]:
        """Get different quantization configurations for V100.

        Returns:
            Dictionary of quantization options with memory estimates
        """
        return {
            "int8": {
                "config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                    llm_int8_threshold=6.0,
                ),
                "estimated_memory_gb": 11.5,  # ~11B params × 1 byte + overhead
                "description": "8-bit quantization with vision modules in fp16",
            },
            "int4": {
                "config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                ),
                "estimated_memory_gb": 6.5,  # ~11B params × 0.5 bytes + overhead
                "description": "4-bit quantization (most aggressive)",
            },
            "mixed_int8_int4": {
                "config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="fp4",
                    llm_int8_skip_modules=[
                        "vision_tower",
                        "multi_modal_projector",
                        "lm_head",
                    ],
                ),
                "estimated_memory_gb": 8.5,
                "description": "Mixed precision with critical layers in higher precision",
            },
            "fp16": {
                "config": None,
                "estimated_memory_gb": 22.0,  # ~11B params × 2 bytes
                "description": "Half precision (won't fit on V100 16GB)",
            },
        }

    def estimate_memory_usage(self, quantization_type: str) -> dict:
        """Estimate memory usage for different configurations.

        Args:
            quantization_type: Type of quantization to use

        Returns:
            Memory usage estimation details
        """
        options = self.get_quantization_options()
        if quantization_type not in options:
            raise ValueError(f"Unknown quantization type: {quantization_type}")

        option = options[quantization_type]
        base_memory = option["estimated_memory_gb"]

        # Add overhead estimates
        overhead_estimates = {
            "kv_cache": 2.0,  # KV cache for sequence generation
            "gradients": 0.0,  # No gradients in inference
            "buffers": 1.5,  # Temporary buffers during computation
            "cuda_overhead": 1.0,  # CUDA kernels and memory fragmentation
        }

        total_memory = base_memory + sum(overhead_estimates.values())

        return {
            "quantization_type": quantization_type,
            "base_model_memory_gb": base_memory,
            "overhead_breakdown": overhead_estimates,
            "total_estimated_gb": total_memory,
            "fits_v100_16gb": total_memory < self.target_memory_gb,
            "safety_margin_gb": self.target_memory_gb - total_memory,
        }

    def load_for_v100(self, quantization_type: str = "int8") -> Tuple[Any, Any]:
        """Load model optimized for V100 16GB.

        Args:
            quantization_type: Quantization strategy ("int8", "int4", "mixed_int8_int4")

        Returns:
            Tuple of (model, processor)
        """
        self.logger.info(
            f"Loading model for V100 16GB with {quantization_type} quantization"
        )

        # Get memory estimate
        memory_estimate = self.estimate_memory_usage(quantization_type)
        self.logger.info(f"Memory estimate: {memory_estimate}")

        if not memory_estimate["fits_v100_16gb"]:
            self.logger.warning(
                f"{quantization_type} may not fit in V100 16GB! "
                f"Estimated: {memory_estimate['total_estimated_gb']:.1f}GB"
            )

        # Clean memory before loading
        self._cleanup_memory()

        # Get quantization config
        options = self.get_quantization_options()
        quantization_config = options[quantization_type]["config"]

        # Load processor
        processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Load model with V100-specific settings
        try:
            # Force CUDA memory efficient attention
            torch.backends.cuda.enable_flash_sdp(False)  # Disable for V100
            torch.backends.cuda.enable_mem_efficient_sdp(True)

            model = MllamaForConditionalGeneration.from_pretrained(
                self.config.model_path,
                device_map="cuda:0",  # Single V100
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                max_memory={0: "15GB"},  # Leave 1GB headroom
                offload_folder="/tmp/offload",  # For emergencies
            )

            # Log actual memory usage
            self._log_gpu_memory_usage()

            # Configure for inference
            model.eval()
            model.config.use_cache = True  # Enable KV cache

            # Set generation config for memory efficiency
            model.generation_config.max_length = 512  # Limit max generation
            model.generation_config.do_sample = False  # Deterministic for consistency

            return model, processor

        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"OOM Error: {e}")
            self.logger.info("Try using 'int4' quantization for V100 16GB")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _cleanup_memory(self) -> None:
        """Aggressive memory cleanup for V100."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()

    def _log_gpu_memory_usage(self) -> None:
        """Log detailed GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3

            self.logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB")
            self.logger.info(f"GPU Memory - Reserved: {reserved:.2f}GB")
            self.logger.info(f"GPU Memory - Peak: {max_allocated:.2f}GB")
            self.logger.info(f"GPU Memory - Free: {16.0 - reserved:.2f}GB")

    def benchmark_quantization_options(self) -> dict:
        """Benchmark different quantization options (without actually loading)."""
        results = {}

        for quant_type, info in self.get_quantization_options().items():
            estimate = self.estimate_memory_usage(quant_type)
            results[quant_type] = {
                "description": info["description"],
                "memory_estimate": estimate,
                "recommended_for_v100": estimate["fits_v100_16gb"],
                "safety_margin": estimate["safety_margin_gb"],
            }

        # Sort by safety margin
        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1]["safety_margin"], reverse=True)
        )

        return sorted_results


def create_v100_optimized_loader(config: LlamaConfig) -> V100ModelLoader:
    """Create a V100-optimized model loader.

    Args:
        config: Llama configuration

    Returns:
        V100ModelLoader instance
    """
    v100_loader = V100ModelLoader(config)

    # Log quantization options
    logger = setup_logging(config.log_level)
    logger.info("V100 16GB Quantization Options:")

    benchmark = v100_loader.benchmark_quantization_options()
    for quant_type, info in benchmark.items():
        if info["recommended_for_v100"]:
            logger.info(f"✅ {quant_type}: {info['description']}")
            logger.info(
                f"   Memory: {info['memory_estimate']['total_estimated_gb']:.1f}GB"
            )
            logger.info(f"   Safety margin: {info['safety_margin']:.1f}GB")
        else:
            logger.warning(f"❌ {quant_type}: {info['description']} - Too large!")

    return v100_loader
