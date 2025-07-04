# V100 16GB Optimization Guide for Llama 3.2-11B Vision

This guide provides specific optimization steps for running the Llama 3.2-11B Vision model on a V100 16GB GPU.

## Your Development & Deployment Environments

Based on your setup:
- **Local Development**: Mac M1 (MPS support)
- **Remote Development**: 2x L40S GPUs (48GB each - plenty of memory) via JupyterHub
- **Production Deployment**: Single V100 16GB (requires CPU offloading) via JupyterHub

## Installation Steps for V100 Optimization

### 1. Core Dependencies

```bash
# Activate your conda environment
conda activate internvl_env

# Core packages (REQUIRED)
pip install accelerate>=0.25.0      # For device mapping and CPU offloading
pip install safetensors>=0.4.0      # Faster loading, lower memory overhead
pip install python-dotenv>=1.0.0    # Environment configuration

# Jupyter kernel (REQUIRED for JupyterHub)
conda install ipykernel             # Or: pip install ipykernel
conda install ipywidgets            # For progress bars in notebooks

# Memory monitoring (RECOMMENDED)
pip install nvidia-ml-py3           # NVIDIA Management Library
pip install gpustat                 # Simple GPU monitoring
pip install nvitop                  # Interactive GPU viewer

# Performance optimization (RECOMMENDED)
pip install pillow-simd             # Faster image processing (replaces Pillow)
pip install datasets>=2.14.0        # Memory-mapped data loading

# 8-bit quantization (RECOMMENDED for V100 16GB)
pip install bitsandbytes>=0.41.0    # Reduces model from 20GB to 11GB
# Note: If installation fails, check CUDA version and install specific version:
# conda list cudatoolkit  # Check your CUDA version
# pip install bitsandbytes-cuda118  # For CUDA 11.8
```

### 2. Register Jupyter Kernel for JupyterHub

```bash
# Make sure you're in the correct conda environment
conda activate internvl_env

# Install ipykernel in the environment
conda install ipykernel

# Register the kernel with JupyterHub
python -m ipykernel install --user --name internvl_env --display-name "InternVL Environment"

# Verify kernel is registered
jupyter kernelspec list
```

You should see output like:
```
Available kernels:
  internvl_env    /home/jovyan/.local/share/jupyter/kernels/internvl_env
  python3         /opt/conda/share/jupyter/kernels/python3
```

### 3. Environment Configuration

Create or update your `.env` file in the project directory:

```bash
# Basic configuration
TAX_INVOICE_NER_BASE_PATH=/home/jovyan/nfs_share/tod/Llama_3.2
TAX_INVOICE_NER_MODEL_PATH=/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision
TAX_INVOICE_NER_DEVICE=auto

# V100 optimization variables (add to .bashrc or set before running)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
```

### 4. Using the Notebook in JupyterHub

1. **Open JupyterHub** in your browser
2. **Navigate** to the notebook file
3. **Open** `llama_11b_package_demo.ipynb`
4. **Select Kernel**: 
   - Click "Kernel" → "Change Kernel" → "InternVL Environment"
   - Or select it from the kernel dropdown (top right)
5. **Verify** the kernel is correct by running:
   ```python
   import sys
   print(sys.executable)
   # Should show: /home/jovyan/.conda/envs/internvl_env/bin/python
   ```

### 5. Device-Specific Configuration

The notebook automatically detects and configures for each environment:

#### Mac M1 (Local Development)
```python
# Automatic detection: MPS device
# Uses fp32 precision for compatibility
# No GPU memory constraints
```

#### 2x L40S (Remote Development via JupyterHub)
```python
# Automatic detection: Multi-GPU
# Balanced model splitting across both GPUs
# ~10GB per GPU (plenty of headroom)
# No CPU offloading needed
```

#### V100 16GB (Production via JupyterHub)
```python
# Automatic detection: Single GPU < 20GB
# CPU offloading enabled automatically
# GPU: ~14GB (inference layers)
# CPU: ~6GB (storage layers)
# Slight performance penalty but functional
```

### 6. Running the Notebook

#### Local Mac M1
```bash
# Traditional Jupyter
jupyter lab llama_11b_package_demo.ipynb
```

#### Remote JupyterHub (L40S or V100)
1. Access JupyterHub URL
2. Open the notebook
3. Select "InternVL Environment" kernel
4. Run all cells

The notebook automatically adapts to the available hardware.

### 7. V100-Specific Optimizations Applied

The notebook automatically applies these optimizations when V100 is detected:

1. **TF32 Enabled**: Faster matrix operations on V100
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   ```

2. **Memory Fraction**: Uses 95% of GPU memory
   ```python
   torch.cuda.set_per_process_memory_fraction(0.95)
   ```

3. **CPU Offloading**: Automatic layer placement
   ```python
   max_memory = {
       0: "14GB",      # GPU (inference-critical layers)
       "cpu": "20GB"   # CPU (storage layers)
   }
   ```

4. **Offload Cache**: Creates local cache directory
   ```python
   offload_folder="./offload_cache"
   ```

### 8. Performance Expectations

| Environment | Memory Usage | Performance | Notes |
|-------------|--------------|-------------|-------|
| Mac M1 | System RAM | Baseline | MPS acceleration helps |
| 2x L40S | ~10GB per GPU | Optimal | No bottlenecks |
| V100 16GB | 14GB GPU + 6GB CPU | ~80-90% of L40S | CPU offloading penalty |

### 9. Monitoring GPU Usage in JupyterHub

Create a new cell in your notebook:
```python
# GPU monitoring cell
import subprocess
import time

def monitor_gpu():
    """Monitor GPU usage"""
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)

# Run once
monitor_gpu()

# Or continuous monitoring (stop with interrupt)
# while True:
#     monitor_gpu()
#     time.sleep(2)
#     print("\n" + "="*80 + "\n")
```

Or from JupyterHub terminal:
```bash
# Open new terminal in JupyterHub
watch -n 1 nvidia-smi
```

### 10. 8-bit Quantization: Why It Reduces Model Size

#### Model Size Calculation

**Current FP16 (16-bit) Model:**
- Llama 3.2-11B has **10.6 billion parameters**
- Each parameter uses **16 bits (2 bytes)** in fp16
- Size: 10.6B × 2 bytes = **~21.2GB** (actual: 19.82GB)

**With 8-bit Quantization:**
- Same **10.6 billion parameters**
- Each parameter uses **8 bits (1 byte)** in int8
- Size: 10.6B × 1 byte = **~10.6GB**
- With overhead (embeddings, norms, metadata): **~11GB total**

#### Benefits for V100 16GB

| Configuration | Model Size | V100 16GB Fit | Speed | Status |
|--------------|------------|---------------|-------|--------|
| FP16 (current) | ~20GB | ❌ Needs CPU offloading | ~80% | Working |
| INT8 (quantized) | ~11GB | ✅ Fits entirely on GPU | 100% | Available |

#### Why Marked as "Future"?

1. **CUDA Compatibility**: Bitsandbytes requires CUDA 11.0+ and specific GPU compute capabilities
2. **Installation Complexity**: May need compilation from source on some systems
3. **Model Support**: Not all model architectures fully support 8-bit inference
4. **Quality Trade-offs**: Minimal (<1% accuracy drop) but needs testing for your use case

#### Enabling 8-bit Quantization NOW

```bash
# Step 1: Install bitsandbytes (may need specific CUDA version)
pip install bitsandbytes>=0.41.0

# If that fails, try:
# pip install bitsandbytes-cuda116  # For CUDA 11.6
# pip install bitsandbytes-cuda117  # For CUDA 11.7
# pip install bitsandbytes-cuda118  # For CUDA 11.8
```

```python
# Step 2: Update model loading code
from transformers import BitsAndBytesConfig

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_quant_type="nf4",  # or "int8"
    bnb_8bit_use_double_quant=True
)

# Load model with quantization
model = MllamaForConditionalGeneration.from_pretrained(
    config['model_path'],
    quantization_config=quantization_config,
    device_map="cuda:0"  # Fits on single V100 now!
)
```

#### Troubleshooting 8-bit Loading

If you encounter issues:

```python
# Check CUDA version
import torch
print(f"CUDA version: {torch.version.cuda}")
print(f"Compute capability: {torch.cuda.get_device_capability()}")

# V100 has compute capability 7.0 - should work!
```

Common issues and fixes:
- **"No kernel image available"**: Wrong CUDA version - reinstall matching bitsandbytes
- **"Module not found"**: Ensure bitsandbytes is installed in correct conda env
- **Quality degradation**: Try `bnb_8bit_quant_type="int8"` instead of "nf4"

### 11. Troubleshooting JupyterHub Issues

#### Kernel Not Showing
```bash
# Re-register the kernel
conda activate internvl_env
python -m ipykernel install --user --name internvl_env --display-name "InternVL Environment" --force
```

#### Wrong Python Environment
```python
# Check in notebook
import sys
print(sys.executable)
print(sys.path)

# Should show internvl_env paths
```

#### Package Import Errors
```bash
# Install packages in the correct environment
conda activate internvl_env
pip install <missing_package>

# Then restart kernel in JupyterHub
```

#### Out of Memory on V100
```python
# Add to notebook before model loading
import torch
import gc

# Clear any existing models
if 'model' in globals():
    del model
gc.collect()
torch.cuda.empty_cache()
```

### 12. Quick Setup Script

Create `setup_v100_kernel.sh`:

```bash
#!/bin/bash
# Quick setup for V100 JupyterHub environment

# Activate environment
conda activate internvl_env

# Install all requirements
pip install accelerate>=0.25.0 safetensors>=0.4.0 python-dotenv>=1.0.0
pip install nvidia-ml-py3 gpustat nvitop
pip install pillow-simd datasets>=2.14.0
conda install ipykernel ipywidgets -y

# Register kernel
python -m ipykernel install --user --name internvl_env --display-name "InternVL Environment"

# List kernels
echo "Available kernels:"
jupyter kernelspec list

echo "Setup complete! Now open JupyterHub and select 'InternVL Environment' kernel"
```

## Summary

Key steps for JupyterHub deployment:
1. Install packages in `internvl_env` conda environment
2. **Register the ipykernel** for JupyterHub
3. Select "InternVL Environment" kernel in notebook
4. The notebook auto-adapts to available hardware (L40S vs V100)

No code changes needed between environments - just select the right kernel!