# Repository Structure

This document describes the cleaned and organized structure of the Llama-3.2-Vision repository.

## 📁 **Current Structure (Clean)**

```
Llama_3.2/
├── 📄 README.md                           # Main documentation
├── 📄 PRODUCTION_CONFIGURATION_GUIDE.md   # Production configuration guide
├── 📄 REPOSITORY_STRUCTURE.md             # This file
├── 📄 prompts.yaml                        # Model prompts configuration
├── 📄 vision_env.yml                      # Conda environment specification
├── 📄 .env                                # Optimized environment variables
├── 📄 llama_package_demo.ipynb            # Main demo notebook
├── 
├── 📂 llama_vision/                       # 🎯 MAIN PACKAGE
│   ├── __init__.py
│   ├── cli/                               # Command-line interfaces
│   │   ├── llama_single.py                # Single image processing
│   │   └── llama_batch.py                 # Batch processing
│   ├── config/                            # Configuration management
│   │   ├── settings.py                    # Environment-based config with .env loading
│   │   └── prompts.py                     # YAML prompt management
│   ├── model/                             # Model loading and inference
│   │   ├── loader.py                      # Model loading with CUDA optimization
│   │   ├── inference.py                   # Inference engine with response cleaning
│   │   └── v100_loader.py                 # V100-specific optimizations
│   ├── extraction/                        # Data extraction modules
│   │   ├── tax_authority_parser.py        # KEY-VALUE and compliance parsing
│   │   ├── key_value_extraction.py        # KEY-VALUE format extraction
│   │   └── json_extraction.py             # JSON format extraction
│   ├── image/                             # Image processing utilities
│   │   ├── loaders.py                     # Image discovery and loading
│   │   └── preprocessing.py               # Image preprocessing
│   ├── evaluation/                        # Evaluation and comparison
│   │   ├── internvl_comparison.py         # InternVL comparison framework
│   │   └── metrics.py                     # Performance metrics
│   └── utils/                             # Utility modules
│       ├── device.py                      # Device detection and optimization
│       └── logging.py                     # Logging utilities
├── 
├── 📂 datasets/                           # Test and training data
│   ├── examples/                          # 🎯 CONSOLIDATED TEST IMAGES
│   │   ├── Costco-petrol.jpg              # Fuel receipt example
│   │   ├── bank_statement_sample.png      # Bank statement example
│   │   ├── invoice.png                    # Invoice example
│   │   └── test_receipt.png               # General receipt example
│   ├── synthetic_receipts/                # Generated receipt data
│   │   ├── images/                        # 100 synthetic receipt images
│   │   ├── metadata.csv                   # Receipt metadata
│   │   └── metadata.json                  # Receipt metadata (JSON)
│   └── synthetic_bank_statements/         # Generated bank statement data
│       └── [8 bank statement samples]
├── 
├── 📂 scripts/                           # Utility scripts (organized)
│   ├── README.md                          # Script documentation
│   ├── debugging/                         # Debug utilities
│   │   ├── debug_llama_vision.py          # Model debugging
│   │   └── model_checks/                  # Model validation scripts
│   ├── deployment/                        # Deployment tools
│   │   ├── v100_deployment_test.py        # V100 deployment testing
│   │   └── quantization/                  # Model quantization scripts
│   ├── setup/                             # Setup and verification
│   │   └── verify_setup.py                # Environment verification
│   └── testing/                           # Test scripts
│       ├── quick_tests/                   # Quick validation tests
│       └── [performance test scripts]
└── 
└── 📂 archive/                           # 🗄️ LEGACY FILES (ignored by git)
    └── legacy/                           # All old/duplicate code moved here
        ├── internvl_PoC/                 # Old InternVL proof-of-concept
        ├── tax_invoice_ner/              # Old package structure
        ├── config/                       # Duplicate configuration files
        ├── examples/                     # Basic example scripts
        ├── docs/                         # Outdated documentation
        ├── tests/                        # Old test structure
        └── results/                      # Test output files
```

## 🎯 **Key Benefits of Clean Structure**

### **Production Ready**
- ✅ Clean package structure (`llama_vision/`)
- ✅ Optimized configuration (`.env` with performance settings)
- ✅ Comprehensive documentation
- ✅ Production configuration guide

### **Organized Assets**
- ✅ Consolidated test images in `datasets/examples/`
- ✅ Well-organized scripts by purpose
- ✅ Clear separation of current vs legacy code

### **Performance Optimized**
- ✅ 75% speed improvement (46s → 11.58s)
- ✅ Enhanced extraction quality (4 → 23+ fields)
- ✅ Response cleaning and artifact removal
- ✅ Australian tax compliance ready

## 📋 **What Was Archived**

All legacy files were moved to `archive/legacy/` (ignored by git):

### **Duplicate Package Structures**
- `tax_invoice_ner/` - Old package name/structure
- `config/` - Duplicate configuration directory
- `examples/` - Basic example scripts (functionality now in package)

### **Development Artifacts**
- `internvl_PoC/` - Large proof-of-concept directory
- `docs/` - Multiple outdated documentation files
- `tests/` - Old test structure
- `results/` - Test output files
- Various configuration and test files

### **Benefits of Archiving**
- 🗂️ **Preserved**: All old code preserved for reference
- 🚀 **Clean**: Repository focused on production-ready code
- 📦 **Smaller**: Significantly reduced repository size
- 🎯 **Clear**: Easy to navigate and understand structure

## 🔧 **Usage After Cleanup**

### **Main Commands (Unchanged)**
```bash
# Single image processing
python -m llama_vision.cli.llama_single extract datasets/examples/Costco-petrol.jpg

# Batch processing  
python -m llama_vision.cli.llama_batch extract datasets/examples/ --output-file results.csv

# Run demo notebook
jupyter notebook llama_package_demo.ipynb
```

### **Configuration**
- **Environment**: `.env` file with optimized performance settings
- **Prompts**: `prompts.yaml` with Australian business patterns
- **Dependencies**: `vision_env.yml` conda environment

### **Documentation**
- **Main**: `README.md` - Complete package documentation
- **Production**: `PRODUCTION_CONFIGURATION_GUIDE.md` - Custom configuration guide
- **Structure**: `REPOSITORY_STRUCTURE.md` - This file

## 🚀 **Next Steps**

1. **Production Deployment**: Use `PRODUCTION_CONFIGURATION_GUIDE.md` to customize for your environment
2. **Performance**: The system is optimized for 75% speed improvement
3. **Scaling**: Use batch processing for multiple documents
4. **Compliance**: Ready for Australian tax authority requirements

The repository is now clean, organized, and production-ready! 🎉