"""Device detection and management utilities."""

import platform
from typing import Any, Dict

import torch


def detect_device() -> Dict[str, Any]:
    """Detect optimal device configuration based on hardware.

    Returns:
        Dictionary with device information
    """
    device_info = {
        "type": "cpu",
        "count": 0,
        "name": "CPU",
        "memory_gb": 0.0,
        "platform": platform.system(),
        "architecture": platform.machine(),
    }

    # Check for CUDA
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_info.update(
            {
                "type": "cuda",
                "count": device_count,
                "name": torch.cuda.get_device_name(0),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory
                / (1024**3),
                "cuda_version": torch.version.cuda,
            }
        )

        # Get info for all GPUs if multiple
        if device_count > 1:
            device_info["devices"] = []
            for i in range(device_count):
                device_info["devices"].append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_gb": torch.cuda.get_device_properties(i).total_memory
                        / (1024**3),
                    }
                )

    # Check for MPS (Mac Metal)
    elif torch.backends.mps.is_available():
        device_info.update(
            {
                "type": "mps",
                "count": 1,
                "name": "Apple Metal Performance Shaders",
                "memory_gb": 0.0,  # MPS uses unified memory
            }
        )

    return device_info


def get_optimal_device_map(device_info: Dict[str, Any]) -> str | Dict[str, int]:
    """Get optimal device mapping based on detected hardware.

    Args:
        device_info: Device information from detect_device()

    Returns:
        Device map string or dictionary
    """
    if device_info["type"] == "cuda":
        if device_info["count"] > 1:
            return "balanced"  # Distribute across multiple GPUs
        else:
            return "cuda:0"  # Single GPU
    elif device_info["type"] == "mps":
        return "mps"
    else:
        return "cpu"


def estimate_memory_requirements(
    model_size: str, use_quantization: bool = False
) -> Dict[str, float]:
    """Estimate memory requirements for different model sizes.

    Args:
        model_size: Model size identifier (e.g., "11B", "1B")
        use_quantization: Whether quantization is used

    Returns:
        Dictionary with memory estimates
    """
    # Base memory requirements in GB (FP16)
    memory_estimates = {
        "1B": {"fp16": 2.0, "int8": 1.0, "recommended": 4.0},
        "3B": {"fp16": 6.0, "int8": 3.0, "recommended": 8.0},
        "7B": {"fp16": 14.0, "int8": 7.0, "recommended": 16.0},
        "11B": {"fp16": 22.0, "int8": 11.0, "recommended": 24.0},
        "13B": {"fp16": 26.0, "int8": 13.0, "recommended": 32.0},
    }

    # Extract size from model identifier (check larger sizes first)
    size_key = "11B"  # Default for Llama-3.2-Vision
    for key in ["13B", "11B", "7B", "3B", "1B"]:  # Check larger sizes first
        if key in model_size.upper():
            size_key = key
            break

    estimates = memory_estimates.get(size_key, memory_estimates["11B"])

    return {
        "model_size": size_key,
        "fp16_memory_gb": estimates["fp16"],
        "int8_memory_gb": estimates["int8"],
        "recommended_vram_gb": estimates["recommended"],
        "estimated_memory_gb": estimates["int8"]
        if use_quantization
        else estimates["fp16"],
        "quantization_enabled": use_quantization,
    }


def check_device_compatibility(
    model_requirements: Dict[str, float], device_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Check if device can handle model requirements.

    Args:
        model_requirements: Memory requirements from estimate_memory_requirements()
        device_info: Device information from detect_device()

    Returns:
        Compatibility assessment
    """
    compatibility = {
        "compatible": False,
        "recommendation": "cpu",
        "memory_sufficient": False,
        "notes": [],
    }

    required_memory = model_requirements["estimated_memory_gb"]

    if device_info["type"] == "cuda":
        available_memory = device_info["memory_gb"]

        if available_memory >= required_memory:
            compatibility.update(
                {
                    "compatible": True,
                    "recommendation": "cuda",
                    "memory_sufficient": True,
                }
            )
            compatibility["notes"].append(
                f"GPU has {available_memory:.1f}GB, requires {required_memory:.1f}GB"
            )
        else:
            compatibility["notes"].append(
                f"GPU has {available_memory:.1f}GB, requires {required_memory:.1f}GB"
            )
            compatibility["notes"].append("Consider enabling quantization or using CPU")

    elif device_info["type"] == "mps":
        # MPS uses unified memory, harder to predict
        compatibility.update(
            {
                "compatible": True,
                "recommendation": "mps",
                "memory_sufficient": True,  # Assume sufficient for now
            }
        )
        compatibility["notes"].append(
            "MPS uses unified memory - monitor system RAM usage"
        )

    else:
        # CPU fallback
        compatibility.update(
            {
                "compatible": True,
                "recommendation": "cpu",
                "memory_sufficient": True,  # Assume system has enough RAM
            }
        )
        compatibility["notes"].append(
            "Using CPU - will be slower but more memory available"
        )

    return compatibility
