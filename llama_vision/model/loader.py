"""Model loading functionality for Llama-3.2-Vision package."""

import gc
import time
from pathlib import Path
from typing import Any, Tuple

import psutil
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
)

from ..config import LlamaConfig
from ..utils import detect_device, setup_logging


class LlamaModelLoader:
    """Load Llama-3.2-Vision model with device optimization."""

    def __init__(self, config: LlamaConfig):
        """Initialize model loader.

        Args:
            config: Llama configuration object
        """
        self.config = config
        self.logger = setup_logging(config.log_level)

        # Enable TF32 for GPU optimization on compatible hardware
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("TF32 enabled for GPU optimization")

    def _get_quantization_config(self) -> BitsAndBytesConfig | None:
        """Get quantization configuration if enabled."""
        if not self.config.use_quantization:
            return None

        try:
            # Check for V100-specific configuration
            if hasattr(self.config, "quantization_type"):
                quant_type = self.config.quantization_type

                if quant_type == "int4":
                    return BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                    )
                elif quant_type == "int8":
                    return BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True,
                        llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                        llm_int8_threshold=6.0,
                    )

            # Default to int8
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_skip_modules=["vision_tower", "mm_projector"],
                llm_int8_threshold=6.0,
            )
        except ImportError:
            self.logger.warning(
                "BitsAndBytesConfig not available - falling back to FP16"
            )
            return None

    def _detect_device_mapping(self) -> str | dict[str, int]:
        """Detect optimal device mapping based on hardware."""
        device_info = detect_device()

        if device_info["type"] == "cuda":
            if device_info["count"] > 1:
                return "balanced"  # Distribute across multiple GPUs
            else:
                return "cuda:0"  # Single GPU
        elif device_info["type"] == "mps":
            return "mps"
        else:
            return "cpu"

    def _get_memory_info(self) -> dict[str, float]:
        """Get current memory usage information."""
        memory_info = {
            "system_memory_gb": psutil.virtual_memory().total / (1024**3),
            "system_memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "system_memory_percent": psutil.virtual_memory().percent,
        }

        if torch.cuda.is_available():
            memory_info.update(
                {
                    "gpu_memory_total_gb": torch.cuda.get_device_properties(
                        0
                    ).total_memory
                    / (1024**3),
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved(0) / (1024**3),
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated(0)
                    / (1024**3),
                }
            )
        elif torch.backends.mps.is_available():
            memory_info.update(
                {
                    "mps_memory_allocated_gb": torch.mps.current_allocated_memory()
                    / (1024**3)
                }
            )

        return memory_info

    def _cleanup_memory(self) -> None:
        """Clean up GPU and system memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def load_model(self) -> Tuple[Any, Any]:
        """Load model and processor with CUDA fixes and optimization.

        Returns:
            Tuple of (model, processor)
        """
        self.logger.info(
            f"Loading Llama-3.2-Vision model from {self.config.model_path}"
        )

        # Verify model path exists
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Clean memory before loading
        self._cleanup_memory()

        # Get device configuration
        device_map = self._detect_device_mapping()
        quantization_config = self._get_quantization_config()

        self.logger.info(f"Device map: {device_map}")
        self.logger.info(
            f"Quantization: {'Enabled' if quantization_config else 'Disabled'}"
        )

        # Record loading metrics
        load_start_time = time.time()
        pre_load_memory = self._get_memory_info()

        try:
            # Load processor first
            self.logger.info("Loading processor...")
            processor = AutoProcessor.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                local_files_only=True,
            )
            self.logger.info("Processor loaded successfully")

            # Load model with appropriate configuration
            self.logger.info("Loading model...")

            # Determine loading strategy based on configuration
            if self.config.device == "cpu" or device_map == "cpu":
                # CPU-only loading
                model = MllamaForConditionalGeneration.from_pretrained(
                    str(model_path),
                    device_map=None,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    quantization_config=quantization_config,
                )
                model = model.to("cpu")
                self.logger.info("Model loaded to CPU")

            else:
                # GPU loading with potential fallback
                try:
                    model = MllamaForConditionalGeneration.from_pretrained(
                        str(model_path),
                        device_map=device_map,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                        quantization_config=quantization_config,
                    )
                    self.logger.info(
                        f"Model loaded to GPU with device_map: {device_map}"
                    )

                except Exception as gpu_error:
                    self.logger.warning(f"GPU loading failed: {gpu_error}")
                    self.logger.info("Falling back to CPU loading")

                    model = MllamaForConditionalGeneration.from_pretrained(
                        str(model_path),
                        device_map=None,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                        quantization_config=quantization_config,
                    )
                    model = model.to("cpu")
                    self.logger.info("Model loaded to CPU as fallback")

            # Test model functionality
            self._test_model_functionality(model, processor)

            # Calculate loading metrics
            load_end_time = time.time()
            post_load_memory = self._get_memory_info()

            self.logger.info(
                f"Model loading completed in {load_end_time - load_start_time:.1f} seconds"
            )
            self._log_memory_usage(pre_load_memory, post_load_memory)

            # Store metadata for inference
            model._llama_config = self.config
            model._device_map = device_map
            model._quantization_enabled = quantization_config is not None

            return model, processor

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self._cleanup_memory()
            raise

    def _test_model_functionality(self, model: Any, processor: Any) -> None:
        """Test basic model functionality."""
        try:
            self.logger.info("Testing model functionality...")

            # Simple text-only test
            test_prompt = "Hello, how are you?"
            test_inputs = processor.tokenizer(test_prompt, return_tensors="pt")

            # Move to appropriate device
            if hasattr(model, "device") and model.device.type != "cpu":
                test_inputs = {k: v.to(model.device) for k, v in test_inputs.items()}

            with torch.no_grad():
                test_outputs = model.generate(
                    **test_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            test_response = processor.decode(
                test_outputs[0][test_inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )

            self.logger.info(f"Model test successful: '{test_response[:50]}...'")

        except Exception as e:
            self.logger.warning(f"Model functionality test failed: {e}")

    def _log_memory_usage(self, pre_memory: dict, post_memory: dict) -> None:
        """Log memory usage before and after loading."""
        if "gpu_memory_allocated_gb" in post_memory:
            gpu_used = post_memory["gpu_memory_allocated_gb"]
            gpu_total = post_memory["gpu_memory_total_gb"]
            self.logger.info(
                f"GPU memory: {gpu_used:.1f}GB allocated / {gpu_total:.1f}GB total"
            )

        system_used = post_memory["system_memory_percent"]
        self.logger.info(f"System memory: {system_used:.1f}% used")
