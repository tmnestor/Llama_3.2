# Image Directory Structure Update

This document summarizes the changes made to update the image discovery logic for the new consolidated `datasets` directory structure.

## Changes Made

### 1. Updated Image Loader (`llama_vision/image/loaders.py`)

**Key Changes:**
- Modified `discover_images()` method to only support `datasets` directory
- Added intelligent path detection for `datasets` directory
- Added support for subdirectories within `datasets`
- Removed legacy directory structure support for simplicity
- Updated test file discovery to only check `datasets` directory

**Directory Discovery Logic:**
```python
# Check for datasets directory structure
datasets_path = None
if base_path.name == "datasets":
    datasets_path = base_path
elif (base_path / "datasets").exists():
    datasets_path = base_path / "datasets"
elif (data_parent / "datasets").exists():
    datasets_path = data_parent / "datasets"
```

**Supported Subdirectories:**
- `datasets/examples`
- `datasets/synthetic_receipts`
- `datasets/synthetic_bank_statements`
- `datasets/test_images`
- `datasets/synthetic_receipts/images`

### 2. Updated Configuration (`llama_vision/config/settings.py`)

**Change:**
- Updated default image path from `/home/jovyan/nfs_share/tod/data/examples` to `/home/jovyan/nfs_share/tod/Llama_3.2/datasets`

### 3. Updated Jupyter Notebook (`llama_package_demo.ipynb`)

**Change:**
- Updated datasets path from `datasets/examples` to `datasets` in cell 10

## Directory Structure Support

### Required Structure
```
Llama_3.2/
├── datasets/
│   ├── examples/
│   │   ├── bank_statement_sample.png
│   │   ├── invoice.png
│   │   ├── test_receipt.png
│   │   └── ...
│   ├── synthetic_receipts/
│   │   ├── images/
│   │   │   ├── receipt_collage_00000.png
│   │   │   └── ...
│   │   ├── metadata.csv
│   │   └── metadata.json
│   └── synthetic_bank_statements/
│       ├── australian_bank_statement_sample_1.png
│       └── ...
```

## Benefits

1. **Consolidated Organization**: All images now in one `datasets` directory
2. **Simplified Logic**: No complex legacy compatibility code
3. **Intelligent Detection**: Automatically finds `datasets` directory at multiple levels
4. **Subdirectory Support**: Properly handles existing subdirectory structure
5. **Clear Requirements**: Users know exactly where to put images

## Image Discovery Categories

The updated system now discovers images in these categories:

1. **`datasets`** - All images in the main datasets directory
2. **`datasets_examples`** - Images in datasets/examples subdirectory
3. **`datasets_synthetic_receipts`** - Images in datasets/synthetic_receipts subdirectory
4. **`datasets_synthetic_bank_statements`** - Images in datasets/synthetic_bank_statements subdirectory
5. **`synthetic_receipts_images`** - Images in datasets/synthetic_receipts/images subdirectory
6. **`test_receipt`** - Specific test files found in various locations

## Files That Still Need Updates

The following files contain hardcoded paths that should be updated for consistency:

### Debug Scripts
- `scripts/debugging/debug_llama_vision.py` - Line 15
- `scripts/debugging/debug_vision_model.py` - Lines 173-176
- `scripts/testing/test_tensor_fix.py` - Line 12

### Shell Scripts
- `scripts/testing/quick_tests/quick_test_bank.sh` - Line 16
- `scripts/testing/quick_tests/quick_test_invoice.sh` - Line 16
- `scripts/testing/quick_tests/test_extraction_gpu.sh` - Lines 18, 27, 36, 52, 80

### Environment File
Update the `.env` file to use the new path:
```bash
LLAMA_VISION_IMAGE_PATH=/home/jovyan/nfs_share/tod/Llama_3.2/datasets
```

## Testing the Update

To test the updated image discovery:

1. **CLI Testing:**
   ```bash
   # Test batch processing with new directory
   llama-batch --image-folder datasets --output-dir output/test
   
   # Test single image
   llama-single --image-path datasets/examples/test_receipt.png
   ```

2. **Python Testing:**
   ```python
   from llama_vision.image import ImageLoader
   
   loader = ImageLoader()
   discovered = loader.discover_images("datasets")
   print(f"Found {sum(len(imgs) for imgs in discovered.values())} images")
   ```

3. **Jupyter Testing:**
   Run the updated `llama_package_demo.ipynb` notebook to verify image discovery works correctly.

## Migration Guide

For existing users:

1. **Move Images**: Move all images from various directories to `datasets/`
2. **Update Environment**: Update `.env` file with new image path
3. **Update Scripts**: Update any custom scripts using hardcoded paths
4. **Test Discovery**: Run image discovery to verify all images are found

**Important**: The system now **only** supports the consolidated `datasets` directory structure. Legacy paths will no longer work.