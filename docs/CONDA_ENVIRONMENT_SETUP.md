# Conda Environment Setup Guide for JupyterHub

This guide provides comprehensive instructions for creating conda environments from `environment.yml` files and registering them as Jupyter kernels for use in JupyterHub environments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding environment.yml](#understanding-environmentyml)
3. [Creating Conda Environments](#creating-conda-environments)
4. [Jupyter Kernel Setup](#jupyter-kernel-setup)
5. [JupyterHub Integration](#jupyterhub-integration)
6. [Managing Multiple Environments](#managing-multiple-environments)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Quick Start

```bash
# 1. Create environment from yml file
conda env create -f environment.yml

# 2. Activate the environment
conda activate myenv

# 3. Install ipykernel
conda install ipykernel

# 4. Register as Jupyter kernel
python -m ipykernel install --user --name myenv --display-name "My Environment"
```

## Understanding environment.yml

### Basic Structure

```yaml
name: myenv
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy=1.24
  - pandas>=2.0
  - pip
  - pip:
    - transformers>=4.36.0
    - torch>=2.0.0
```

### Complete Example for ML/NLP Project

```yaml
name: llama_vision_env
channels:
  - conda-forge
  - pytorch
  - nvidia
  - defaults
dependencies:
  # Python version
  - python=3.11
  
  # Core scientific packages
  - numpy=1.24.*
  - pandas>=2.0.0
  - scikit-learn>=1.3.0
  
  # Deep learning
  - pytorch>=2.0.0
  - torchvision
  - torchaudio
  - cudatoolkit=11.8  # For GPU support
  
  # Jupyter
  - ipykernel
  - ipywidgets
  - notebook
  
  # Development tools
  - pytest
  - black
  - ruff
  
  # Pip dependencies
  - pip
  - pip:
    # Transformers ecosystem
    - transformers>=4.36.0
    - accelerate>=0.25.0
    - datasets>=2.14.0
    - safetensors>=0.4.0
    
    # Utilities
    - python-dotenv
    - rich
    - typer
    
    # Optional optimizations
    - bitsandbytes>=0.41.0
    - pillow-simd
```

## Creating Conda Environments

### From environment.yml

```bash
# Create environment from file
conda env create -f environment.yml

# Create with custom name
conda env create -f environment.yml -n custom_name

# Create in specific location
conda env create -f environment.yml -p /path/to/env
```

### Updating Existing Environment

```bash
# Update environment with new packages
conda env update -f environment.yml

# Update and prune removed packages
conda env update -f environment.yml --prune
```

### Exporting Environment

```bash
# Export current environment to yml
conda env export > environment.yml

# Export without build strings (more portable)
conda env export --no-builds > environment.yml

# Export only explicitly installed packages
conda env export --from-history > environment.yml
```

## Jupyter Kernel Setup

### Installing IPython Kernel

```bash
# Activate your environment first
conda activate myenv

# Install ipykernel in the environment
conda install ipykernel

# Or via pip
pip install ipykernel
```

### Registering Kernel for JupyterHub

```bash
# Basic registration
python -m ipykernel install --user --name myenv

# With custom display name
python -m ipykernel install --user --name myenv --display-name "My Project Environment"

# With specific path (for shared environments)
python -m ipykernel install --prefix=/opt/conda --name myenv --display-name "Shared Environment"
```

### Kernel Specification

The kernel spec is created at:
- User: `~/.local/share/jupyter/kernels/myenv/`
- System: `/usr/local/share/jupyter/kernels/myenv/`

Example `kernel.json`:
```json
{
 "argv": [
  "/home/user/.conda/envs/myenv/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "My Environment",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
```

## JupyterHub Integration

### For Individual Users

```bash
# 1. Create and activate environment
conda env create -f environment.yml
conda activate myenv

# 2. Install kernel packages
conda install ipykernel ipywidgets

# 3. Register kernel
python -m ipykernel install --user --name myenv --display-name "Project Environment"

# 4. Verify installation
jupyter kernelspec list
```

### For System-wide Deployment

```bash
# As admin/root user
# 1. Create environment in shared location
conda env create -f environment.yml -p /opt/conda/envs/shared_env

# 2. Activate environment
conda activate /opt/conda/envs/shared_env

# 3. Install kernel
/opt/conda/envs/shared_env/bin/python -m ipykernel install \
    --prefix=/opt/conda \
    --name shared_env \
    --display-name "Shared ML Environment"
```

### Docker-based JupyterHub

```dockerfile
# Dockerfile example
FROM jupyter/base-notebook

# Copy environment file
COPY environment.yml /tmp/

# Create conda environment
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Install kernel
RUN /opt/conda/envs/myenv/bin/python -m ipykernel install \
    --name myenv \
    --display_name "Custom Environment"

# Set default kernel
ENV JUPYTER_ENABLE_LAB=yes
```

## Managing Multiple Environments

### Project Structure

```
project/
├── environments/
│   ├── dev-environment.yml      # Development environment
│   ├── prod-environment.yml     # Production environment
│   └── test-environment.yml     # Testing environment
├── notebooks/
├── src/
└── setup_kernels.sh            # Automation script
```

### Automation Script

```bash
#!/bin/bash
# setup_kernels.sh - Setup all project kernels

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Function to setup environment and kernel
setup_kernel() {
    local env_file=$1
    local env_name=$2
    local display_name=$3
    
    echo -e "${GREEN}Setting up $display_name...${NC}"
    
    # Create environment
    conda env create -f $env_file -n $env_name --force
    
    # Install kernel
    conda run -n $env_name python -m ipykernel install \
        --user \
        --name $env_name \
        --display-name "$display_name"
    
    echo -e "${GREEN}✓ $display_name setup complete${NC}"
}

# Setup environments
setup_kernel "environments/dev-environment.yml" "myproject_dev" "MyProject (Dev)"
setup_kernel "environments/prod-environment.yml" "myproject_prod" "MyProject (Prod)"
setup_kernel "environments/test-environment.yml" "myproject_test" "MyProject (Test)"

# List all kernels
echo -e "\n${GREEN}Available kernels:${NC}"
jupyter kernelspec list
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Kernel Not Showing in JupyterHub

```bash
# Check if kernel is installed
jupyter kernelspec list

# Reinstall kernel
python -m ipykernel install --user --name myenv --force

# Restart JupyterHub server
```

#### 2. Wrong Python Version in Kernel

```python
# Check in notebook
import sys
print(sys.executable)
print(sys.version)

# Should show path to conda environment
```

#### 3. Package Import Errors

```bash
# Ensure packages are installed in correct environment
conda activate myenv
conda list  # Verify packages

# Reinstall missing packages
conda install missing_package
# or
pip install missing_package
```

#### 4. Permission Errors

```bash
# For user installation
python -m ipykernel install --user --name myenv

# For system-wide (requires sudo/admin)
sudo python -m ipykernel install --name myenv
```

#### 5. Kernel Crashes

```bash
# Check kernel log
jupyter kernelspec list  # Find kernel path
cat ~/.local/share/jupyter/kernels/myenv/kernel.json

# Verify Python path is correct
/path/to/python -m ipykernel_launcher --help
```

### Debugging Commands

```bash
# List all conda environments
conda env list

# Show environment details
conda info --envs

# Check kernel specifications
jupyter kernelspec list

# Remove a kernel
jupyter kernelspec uninstall myenv

# Test kernel connection
jupyter console --kernel=myenv
```

## Best Practices

### 1. Environment Naming Convention

```yaml
# Use descriptive names
name: projectname_purpose_pyversion

# Examples:
name: llama_dev_py311
name: mlops_prod_py310
name: datascience_test_py39
```

### 2. Version Pinning

```yaml
dependencies:
  # Pin major versions for stability
  - python=3.11.*
  - numpy=1.24.*
  
  # Use >= for flexibility
  - pandas>=2.0.0,<3.0.0
  
  # Exact versions for production
  - torch==2.0.1
```

### 3. Channel Priority

```yaml
# Order matters - specific channels first
channels:
  - conda-forge
  - pytorch
  - nvidia
  - defaults
  
channel_priority: strict  # Optional: enforce channel priority
```

### 4. Kernel Metadata

```bash
# Add metadata to help users
python -m ipykernel install \
    --user \
    --name llama_v100 \
    --display-name "Llama 3.2 (V100 Optimized)" \
    --env CUDA_VISIBLE_DEVICES 0 \
    --env PYTORCH_CUDA_ALLOC_CONF max_split_size_mb:512
```

### 5. Documentation

Create a `KERNELS.md` file:

```markdown
# Available Jupyter Kernels

## Llama Vision Environment
- **Name**: `llama_vision_env`
- **Display**: "Llama 3.2 Vision"
- **Python**: 3.11
- **Key Packages**: torch, transformers, accelerate
- **GPU**: CUDA 11.8 support
- **Use Case**: Running Llama 3.2-11B Vision model

## Data Science Environment
- **Name**: `datascience_env`
- **Display**: "Data Science Stack"
- **Python**: 3.10
- **Key Packages**: pandas, scikit-learn, matplotlib
- **Use Case**: General data analysis
```

## Quick Reference

### Essential Commands

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate myenv

# Install Jupyter kernel
conda install ipykernel
python -m ipykernel install --user --name myenv --display-name "My Env"

# List kernels
jupyter kernelspec list

# Remove kernel
jupyter kernelspec uninstall myenv

# Update environment
conda env update -f environment.yml --prune

# Export environment
conda env export --from-history > environment.yml

# Remove environment
conda env remove -n myenv
```

### Environment Variables for Kernels

```bash
# GPU-specific kernel
python -m ipykernel install \
    --user \
    --name gpu_env \
    --display-name "GPU Environment" \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Conclusion

Proper conda environment and Jupyter kernel management ensures:
- **Reproducible** research and development
- **Isolated** dependencies per project
- **Easy** switching between environments in JupyterHub
- **Consistent** experience across team members

Remember to document your environments and keep `environment.yml` files in version control!