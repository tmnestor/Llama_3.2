# Environment Configuration Guide (.env)

This document explains the configuration decisions and optimal settings for different hardware configurations in the Llama-3.2-Vision package.

## 📋 Table of Contents

- [Overview](#overview)
- [Configuration Categories](#configuration-categories)
- [Hardware-Specific Configurations](#hardware-specific-configurations)
- [Performance Tuning](#performance-tuning)
- [Configuration Examples](#configuration-examples)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

The `.env` file controls all aspects of the Llama-3.2-Vision package behavior, from model loading to inference optimization. Configuration decisions are based on:

- **Hardware capabilities** (GPU VRAM, compute architecture)
- **Performance requirements** (throughput vs. accuracy)
- **Memory constraints** (available system and GPU memory)
- **Parallelization strategy** (I/O vs. compute workers)

## 📊 Configuration Categories

### 1. **Path Configuration**
```bash
# Project structure paths
LLAMA_VISION_BASE_PATH=/home/jovyan/nfs_share/tod/Llama_3.2
LLAMA_VISION_MODEL_PATH=/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision
LLAMA_VISION_IMAGE_PATH=/home/jovyan/nfs_share/tod/Llama_3.2/datasets
LLAMA_VISION_OUTPUT_PATH=/home/jovyan/nfs_share/tod/Llama_3.2/output
```

**Decision Rationale:**
- **Centralized structure**: All project files under single base path
- **Shared model storage**: Models stored in `/models/` for multi-project access
- **Consolidated datasets**: Images in `/datasets/` directory for simplified discovery
- **Organized output**: Results in `/output/` to separate from source code

### 2. **Model Configuration**
```bash
# Core model settings
LLAMA_VISION_DEVICE=cuda                    # Compute device
LLAMA_VISION_USE_8BIT=false                # Quantization toggle
LLAMA_VISION_MAX_TOKENS=256                # Generation length
LLAMA_VISION_TEMPERATURE=0.0               # Randomness control
LLAMA_VISION_DO_SAMPLE=false               # Sampling toggle
```

**Decision Rationale:**

#### **Device Selection (`LLAMA_VISION_DEVICE=cuda`)**
- **Why CUDA**: Leverages GPU acceleration for 10-100x speedup over CPU
- **Alternatives**: `cpu` (fallback), `mps` (Apple Silicon)
- **Auto-detection**: Could use `auto` but explicit is more predictable

#### **Quantization (`LLAMA_VISION_USE_8BIT=false`)**
- **Why Disabled**: H200 has 80GB VRAM, sufficient for full precision
- **When to Enable**: GPU < 24GB VRAM, or when maximizing throughput over quality
- **Trade-offs**: 8-bit saves ~50% memory but reduces accuracy by 2-5%

#### **Token Limits (`LLAMA_VISION_MAX_TOKENS=256`)**
- **Why 256**: Sufficient for structured extraction, prevents repetition
- **Too Low**: < 128 tokens may truncate complex extractions
- **Too High**: > 512 tokens waste compute and increase repetition risk

#### **Temperature (`LLAMA_VISION_TEMPERATURE=0.0`)**
- **Why 0.0**: Deterministic output for consistent extraction
- **Alternative**: 0.1-0.3 for slight creativity in edge cases
- **Avoid**: > 0.5 introduces too much randomness for structured tasks

#### **Sampling (`LLAMA_VISION_DO_SAMPLE=false`)**
- **Why Disabled**: Greedy decoding ensures consistency
- **When to Enable**: Temperature > 0.0 and need creativity
- **Impact**: Sampling adds variability but reduces reliability

### 3. **Memory and Performance Settings**
```bash
# Inference optimization
LLAMA_VISION_CLASSIFICATION_MAX_TOKENS=20   # Classification response length
LLAMA_VISION_EXTRACTION_MAX_TOKENS=256      # Extraction response length
LLAMA_VISION_MEMORY_CLEANUP_ENABLED=false  # Memory cleanup toggle
LLAMA_VISION_PROCESS_BATCH_SIZE=4          # Batch processing size
LLAMA_VISION_MEMORY_CLEANUP_DELAY=0        # Cleanup delay (seconds)
```

**Decision Rationale:**

#### **Classification Tokens (`LLAMA_VISION_CLASSIFICATION_MAX_TOKENS=20`)**
- **Why 20**: Classification needs only document type (5-15 tokens)
- **Performance**: Shorter generation = faster classification
- **Sufficient for**: "fuel_receipt", "bank_statement", "tax_invoice", "receipt"

#### **Extraction Tokens (`LLAMA_VISION_EXTRACTION_MAX_TOKENS=256`)**
- **Why 256**: Structured JSON extraction requires more tokens
- **Balance**: Long enough for complete extraction, short enough to prevent repetition
- **Typical usage**: 80-180 tokens for most documents

#### **Memory Cleanup (`LLAMA_VISION_MEMORY_CLEANUP_ENABLED=false`)**
- **Why Disabled**: H200 has abundant VRAM, cleanup adds overhead
- **When to Enable**: GPU < 16GB VRAM or processing large batches
- **Impact**: Cleanup reduces memory usage but adds 100-200ms per image

#### **Batch Size (`LLAMA_VISION_PROCESS_BATCH_SIZE=4`)**
- **Why 4**: Optimal balance for H200 between throughput and memory
- **Calculation**: Based on model size (22GB) and available VRAM (80GB)
- **Scaling**: Can increase to 8 if processing simple documents

### 4. **Inference Optimization Settings**
```bash
# Automatic model optimizations (implemented in code)
# model.eval()                              # Evaluation mode for inference-only
# torch.no_grad()                          # Disable gradient computation
# early_stopping_threshold=0.85            # Classification early stopping
```

**Automatic Optimizations:**
- **`model.eval()`**: Disables dropout and batch norm updates for consistent inference
- **`torch.no_grad()`**: Prevents gradient computation during inference
- **Early stopping**: Stops classification when confidence ≥ 85%

### 5. **Parallel Processing Configuration**
```bash
# Parallelization settings
LLAMA_VISION_IMAGE_LOADER_WORKERS=8        # File I/O threads
LLAMA_VISION_MAX_CONCURRENT_IMAGES=16      # Memory buffer size
LLAMA_VISION_ENABLE_PARALLEL_LOADING=true # Parallel loading toggle
```

**Decision Rationale:**

#### **Image Loader Workers (`LLAMA_VISION_IMAGE_LOADER_WORKERS=8`)**
- **Why 8**: Optimal for NFS/network storage with multiple GPUs
- **I/O Bound**: Image loading is I/O bottleneck, not compute
- **Scaling**: Can increase to 16 for faster storage systems

#### **Concurrent Images (`LLAMA_VISION_MAX_CONCURRENT_IMAGES=16`)**
- **Why 16**: 2x buffer vs. processing batch size (4)
- **Memory Cost**: ~50MB per image in memory buffer
- **Purpose**: Keeps model fed while loading next batch

#### **Parallel Loading (`LLAMA_VISION_ENABLE_PARALLEL_LOADING=true`)**
- **Why Enabled**: Dramatically improves I/O performance
- **Impact**: 3-5x faster image discovery and loading
- **Disable only**: For debugging or very limited systems

### 6. **GPU-Specific Optimizations**
```bash
# H200 GPU optimizations
CUDA_VISIBLE_DEVICES=0,1                   # Use both GPUs
TORCH_CUDA_ARCH_LIST=9.0                   # Hopper architecture
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory allocation
NCCL_P2P_DISABLE=0                        # GPU-to-GPU communication
NCCL_IB_DISABLE=0                         # InfiniBand support
```

**Decision Rationale:**

#### **Visible Devices (`CUDA_VISIBLE_DEVICES=0,1`)**
- **Why Both**: Utilizes both H200 GPUs for maximum throughput
- **Model Sharding**: Llama-3.2-11B can be split across GPUs
- **Alternatives**: `0` for single GPU, `1` for second GPU only

#### **Architecture (`TORCH_CUDA_ARCH_LIST=9.0`)**
- **Why 9.0**: Hopper architecture (H200) compute capability
- **Optimization**: Enables architecture-specific optimizations
- **Other Values**: `8.9` (L40S), `7.0` (V100), `8.0` (A100)

#### **Memory Allocation (`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`)**
- **Why 512MB**: Optimal chunk size for H200's 80GB VRAM
- **Prevents**: Memory fragmentation in long-running processes
- **Alternatives**: `256` (smaller GPUs), `1024` (very large models)

#### **NCCL Settings (`NCCL_P2P_DISABLE=0`, `NCCL_IB_DISABLE=0`)**
- **Why Enabled**: Optimizes multi-GPU communication
- **P2P**: Peer-to-peer GPU communication
- **InfiniBand**: High-speed interconnect (if available)

## 🖥️ Hardware-Specific Configurations

### **H200 (80GB VRAM × 2) - Maximum Performance**
```bash
# Model configuration
LLAMA_VISION_USE_8BIT=false
LLAMA_VISION_PROCESS_BATCH_SIZE=4
LLAMA_VISION_MAX_TOKENS=256

# Parallelization
LLAMA_VISION_IMAGE_LOADER_WORKERS=8
LLAMA_VISION_MAX_CONCURRENT_IMAGES=16
--max-workers 8  # CLI parameter

# GPU optimization
CUDA_VISIBLE_DEVICES=0,1
TORCH_CUDA_ARCH_LIST=9.0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**Expected Performance**: 85-130 images/minute  
**Optimizations Applied**: `model.eval()` + `torch.no_grad()` + early stopping + dual GPU

### **L40S (48GB VRAM) - High Performance**
```bash
# Model configuration
LLAMA_VISION_USE_8BIT=false
LLAMA_VISION_PROCESS_BATCH_SIZE=2
LLAMA_VISION_MAX_TOKENS=256

# Parallelization
LLAMA_VISION_IMAGE_LOADER_WORKERS=6
LLAMA_VISION_MAX_CONCURRENT_IMAGES=8
--max-workers 4  # CLI parameter

# GPU optimization
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=8.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**Expected Performance**: 45-70 images/minute  
**Optimizations Applied**: `model.eval()` + `torch.no_grad()` + early stopping

### **V100 (16GB VRAM) - Balanced Performance**
```bash
# Model configuration
LLAMA_VISION_USE_8BIT=true  # Required for memory
LLAMA_VISION_PROCESS_BATCH_SIZE=1
LLAMA_VISION_MAX_TOKENS=256

# Parallelization
LLAMA_VISION_IMAGE_LOADER_WORKERS=4
LLAMA_VISION_MAX_CONCURRENT_IMAGES=4
--max-workers 2  # CLI parameter

# GPU optimization
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=7.0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

**Expected Performance**: 18-30 images/minute  
**Optimizations Applied**: `model.eval()` + `torch.no_grad()` + early stopping + 8-bit quantization

### **RTX 3090 (24GB VRAM) - Consumer GPU**
```bash
# Model configuration
LLAMA_VISION_USE_8BIT=false
LLAMA_VISION_PROCESS_BATCH_SIZE=1
LLAMA_VISION_MAX_TOKENS=256

# Parallelization
LLAMA_VISION_IMAGE_LOADER_WORKERS=4
LLAMA_VISION_MAX_CONCURRENT_IMAGES=4
--max-workers 2  # CLI parameter

# GPU optimization
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=8.6
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

**Expected Performance**: 24-36 images/minute  
**Optimizations Applied**: `model.eval()` + `torch.no_grad()` + early stopping

### **CPU Only - Fallback Mode**
```bash
# Model configuration
LLAMA_VISION_DEVICE=cpu
LLAMA_VISION_USE_8BIT=false
LLAMA_VISION_PROCESS_BATCH_SIZE=1
LLAMA_VISION_MAX_TOKENS=256

# Parallelization
LLAMA_VISION_IMAGE_LOADER_WORKERS=2
LLAMA_VISION_MAX_CONCURRENT_IMAGES=2
--max-workers 1  # CLI parameter

# No GPU optimization needed
```

**Expected Performance**: 1.2-3.5 images/minute  
**Optimizations Applied**: `model.eval()` + `torch.no_grad()` + early stopping

## 🎯 Performance Tuning

### **Memory Optimization**
```bash
# For memory-constrained systems
LLAMA_VISION_USE_8BIT=true
LLAMA_VISION_MEMORY_CLEANUP_ENABLED=true
LLAMA_VISION_PROCESS_BATCH_SIZE=1
LLAMA_VISION_MAX_CONCURRENT_IMAGES=2
```

### **Throughput Optimization**
```bash
# For maximum processing speed
LLAMA_VISION_PROCESS_BATCH_SIZE=8
LLAMA_VISION_IMAGE_LOADER_WORKERS=16
LLAMA_VISION_MAX_CONCURRENT_IMAGES=32
LLAMA_VISION_MEMORY_CLEANUP_ENABLED=false
```

### **Accuracy Optimization**
```bash
# For maximum extraction quality
LLAMA_VISION_USE_8BIT=false
LLAMA_VISION_TEMPERATURE=0.0
LLAMA_VISION_DO_SAMPLE=false
LLAMA_VISION_MAX_TOKENS=512
```

### **Balanced Configuration**
```bash
# For production use
LLAMA_VISION_PROCESS_BATCH_SIZE=2
LLAMA_VISION_IMAGE_LOADER_WORKERS=4
LLAMA_VISION_MAX_CONCURRENT_IMAGES=8
LLAMA_VISION_MEMORY_CLEANUP_ENABLED=true
```

## 📋 Configuration Examples

### **Development Environment**
```bash
# Local development on workstation
LLAMA_VISION_MODEL_PATH=/home/user/models/Llama-3.2-11B-Vision
LLAMA_VISION_IMAGE_PATH=/home/user/projects/datasets
LLAMA_VISION_LOG_LEVEL=DEBUG
LLAMA_VISION_PROCESS_BATCH_SIZE=1
LLAMA_VISION_IMAGE_LOADER_WORKERS=2
```

### **Production Environment**
```bash
# Remote GPU server
LLAMA_VISION_MODEL_PATH=/shared/models/Llama-3.2-11B-Vision
LLAMA_VISION_IMAGE_PATH=/data/input/documents
LLAMA_VISION_OUTPUT_PATH=/data/output/results
LLAMA_VISION_LOG_LEVEL=INFO
LLAMA_VISION_PROCESS_BATCH_SIZE=4
LLAMA_VISION_IMAGE_LOADER_WORKERS=8
```

### **Batch Processing Environment**
```bash
# High-throughput batch processing
LLAMA_VISION_PROCESS_BATCH_SIZE=8
LLAMA_VISION_MAX_CONCURRENT_IMAGES=32
LLAMA_VISION_IMAGE_LOADER_WORKERS=16
LLAMA_VISION_MEMORY_CLEANUP_ENABLED=false
LLAMA_VISION_CLASSIFICATION_MAX_TOKENS=10
```

## 🔧 Troubleshooting

### **Common Issues and Solutions**

#### **1. Out of Memory Errors**
```bash
# Reduce memory usage
LLAMA_VISION_USE_8BIT=true
LLAMA_VISION_PROCESS_BATCH_SIZE=1
LLAMA_VISION_MAX_CONCURRENT_IMAGES=2
LLAMA_VISION_MEMORY_CLEANUP_ENABLED=true
```

#### **2. Slow Processing**
```bash
# Increase parallelization
LLAMA_VISION_IMAGE_LOADER_WORKERS=8
LLAMA_VISION_MAX_CONCURRENT_IMAGES=16
--max-workers 4
```

#### **3. Model Loading Errors**
```bash
# Check model path and permissions
LLAMA_VISION_MODEL_PATH=/correct/path/to/model
LLAMA_VISION_DEVICE=cuda  # or cpu if GPU unavailable
```

#### **4. Inconsistent Results**
```bash
# Ensure deterministic settings
LLAMA_VISION_TEMPERATURE=0.0
LLAMA_VISION_DO_SAMPLE=false
LLAMA_VISION_USE_8BIT=false
```

#### **5. GPU Not Detected**
```bash
# Force GPU usage
CUDA_VISIBLE_DEVICES=0
LLAMA_VISION_DEVICE=cuda

# Check availability
python -c "import torch; print(torch.cuda.is_available())"
```

### **Performance Monitoring**
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor memory usage
python -c "import torch; print(f'GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB')"

# Monitor processing speed
python -m llama_vision.cli.llama_batch --verbose
```

## 🎚️ Configuration Decision Tree

### **Step 1: Determine GPU Configuration**
```
Do you have GPU access?
├── Yes → What GPU type?
│   ├── H200 (80GB) → Use H200 config
│   ├── L40S (48GB) → Use L40S config
│   ├── V100 (16GB) → Use V100 config
│   └── RTX 3090 (24GB) → Use RTX 3090 config
└── No → Use CPU config
```

### **Step 2: Choose Performance Profile**
```
What's your priority?
├── Maximum Speed → Throughput optimization
├── Memory Efficiency → Memory optimization
├── Best Accuracy → Accuracy optimization
└── Balanced → Balanced configuration
```

### **Step 3: Set Parallelization**
```
Based on your hardware:
├── High-end (H200, L40S) → 8-16 workers
├── Mid-range (V100, 3090) → 4-8 workers
├── Low-end (< 16GB) → 2-4 workers
└── CPU only → 1-2 workers
```

### **Step 4: Configure Memory**
```
Available GPU memory:
├── > 40GB → No quantization, large batches
├── 20-40GB → No quantization, medium batches
├── 10-20GB → Consider quantization
└── < 10GB → Enable quantization, small batches
```

## 🔍 Environment Variables Reference

### **Complete Variable List**
```bash
# Paths
LLAMA_VISION_BASE_PATH           # Project root directory
LLAMA_VISION_MODEL_PATH          # Model files location
LLAMA_VISION_IMAGE_PATH          # Input images directory
LLAMA_VISION_OUTPUT_PATH         # Output results directory
LLAMA_VISION_CONFIG_PATH         # Configuration file path

# Model Settings
LLAMA_VISION_DEVICE              # cuda/cpu/mps
LLAMA_VISION_USE_8BIT            # true/false
LLAMA_VISION_MAX_TOKENS          # 128-1024
LLAMA_VISION_TEMPERATURE         # 0.0-1.0
LLAMA_VISION_DO_SAMPLE           # true/false
LLAMA_VISION_TOP_P              # 0.1-1.0
LLAMA_VISION_TOP_K              # 1-100
LLAMA_VISION_PAD_TOKEN_ID       # -1 (auto)

# Performance
LLAMA_VISION_CLASSIFICATION_MAX_TOKENS  # 10-50
LLAMA_VISION_EXTRACTION_MAX_TOKENS      # 256-1024
LLAMA_VISION_MEMORY_CLEANUP_ENABLED     # true/false
LLAMA_VISION_PROCESS_BATCH_SIZE         # 1-8
LLAMA_VISION_MEMORY_CLEANUP_DELAY       # 0-5 seconds

# Parallelization
LLAMA_VISION_IMAGE_LOADER_WORKERS       # 2-16
LLAMA_VISION_MAX_CONCURRENT_IMAGES      # 2-32
LLAMA_VISION_ENABLE_PARALLEL_LOADING    # true/false

# GPU Optimization
CUDA_VISIBLE_DEVICES                    # 0,1 or 0 or 1
TORCH_CUDA_ARCH_LIST                    # 7.0/8.6/8.9/9.0
PYTORCH_CUDA_ALLOC_CONF                 # memory config
NCCL_P2P_DISABLE                        # 0/1
NCCL_IB_DISABLE                         # 0/1

# Application
LLAMA_VISION_ENVIRONMENT                # local/production
LLAMA_VISION_LOG_LEVEL                  # DEBUG/INFO/WARNING/ERROR
LLAMA_VISION_ENABLE_METRICS             # true/false
LLAMA_VISION_ENABLE_ABN_VALIDATION      # true/false
LLAMA_VISION_ENABLE_GST_VALIDATION      # true/false
LLAMA_VISION_DEFAULT_CURRENCY           # AUD/USD/EUR
LLAMA_VISION_DATE_FORMAT                # DD/MM/YYYY
```

## 🎯 Best Practices

### **1. Configuration Management**
- **Version control**: Keep `.env` in git with sensitive data removed
- **Environment-specific**: Use different `.env` files for dev/prod
- **Documentation**: Comment complex configurations
- **Testing**: Validate configurations before deployment

### **2. Performance Monitoring**
- **Baseline**: Measure performance with default settings
- **Incremental**: Change one parameter at a time
- **Validation**: Ensure accuracy isn't sacrificed for speed
- **Monitoring**: Track GPU utilization and memory usage

### **3. Hardware Considerations**
- **Memory**: Always leave 10-20% GPU memory free
- **Cooling**: Monitor GPU temperatures during heavy processing
- **Power**: Ensure adequate power supply for high-end GPUs
- **Storage**: Use fast storage (NVMe/SSD) for datasets

### **4. Deployment Guidelines**
- **Staging**: Test configurations in staging environment
- **Rollback**: Keep working configurations as backup
- **Monitoring**: Set up alerts for performance degradation
- **Documentation**: Document configuration changes and rationale

---

*This guide provides comprehensive configuration guidance for optimal Llama-3.2-Vision performance across different hardware configurations.*