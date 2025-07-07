# Scripts Directory Organization

This directory contains utility scripts organized by purpose.

## Directory Structure

```
scripts/
├── deployment/          # Deployment and optimization scripts
│   ├── v100_setup.py   # V100 GPU deployment setup
│   └── quantization/   # Quantization analysis and testing
│
├── testing/            # Test scripts and quick tests
│   ├── integration/    # Integration tests
│   └── quick_tests/    # Quick test scripts
│
├── debugging/          # Debug and troubleshooting scripts
│   └── model_checks/   # Model verification scripts
│
├── legacy/             # Legacy scripts (to be migrated/removed)
│   └── archive/        # Archived old scripts
│
└── setup/              # Setup and verification scripts
    └── verify_setup.py # Environment verification
```

## Script Categories

### 🚀 Deployment Scripts (`deployment/`)
- V100 optimization and setup
- Quantization analysis
- Memory profiling
- Performance benchmarking

### 🧪 Testing Scripts (`testing/`)
- Quick test scripts for different components
- Integration tests
- Model inference tests
- Extraction accuracy tests

### 🔍 Debugging Scripts (`debugging/`)
- Model loading diagnostics
- Memory leak detection
- CUDA error troubleshooting
- Performance bottleneck analysis

### 📦 Setup Scripts (`setup/`)
- Environment verification
- Dependency checks
- Configuration validation

### 🗄️ Legacy Scripts (`legacy/`)
- Old notebook-based scripts
- Deprecated utilities
- Scripts to be migrated to new structure

## Usage Guidelines

1. **Naming Convention**:
   - Use descriptive names: `test_<component>.py`, `debug_<issue>.py`
   - Avoid generic names like `test.py` or `fix.py`

2. **Documentation**:
   - Each script should have a docstring explaining its purpose
   - Include usage examples in the docstring

3. **Dependencies**:
   - Scripts should work with the conda environment
   - Avoid adding script-specific dependencies

4. **Organization**:
   - Place scripts in appropriate subdirectories
   - Move legacy scripts to `legacy/` before removal