# Adapting to Llama-3.2-11B-Vision Model

This guide explains how to adapt the current 1B Vision implementation to use the official **meta-llama/Llama-3.2-11B-Vision** model for production deployment on remote GPU hosts.

## Overview

The current codebase is configured for local development using `kadirnar/Llama-3.2-1B-Vision` (custom Llamavision architecture). This document outlines the changes needed to switch to the official **meta-llama/Llama-3.2-11B-Vision** model for production use.

## Key Differences Between Models

| Aspect | 1B Model (Current) | 11B Model (Target) |
|--------|-------------------|-------------------|
| **Model Size** | 1B parameters | 11B parameters |
| **Architecture** | Custom Llamavision | Official LlamaForCausalLM |
| **Memory Required** | ~4GB VRAM | ~22GB VRAM (8-bit) / ~44GB VRAM (16-bit) |
| **Interface** | `answer_question()` method | Standard transformers chat interface |
| **Performance** | Basic extraction | Production-quality extraction |
| **Use Case** | Local development/prototyping | Production deployment |

## Required Changes

### 1. Model Path Configuration

Update model paths in configuration files and scripts:

**File: `config/extractor/llama_vision_config.yaml`**
```yaml
model:
  # Change from:
  model_path: "/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision"
  
  # To:
  model_path: "/efs/shared/models/Llama-3.2-11B-Vision"
  # OR for local testing (if you have sufficient VRAM):
  # model_path: "/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision"
```

**File: `models/extractors/llama_vision_extractor.py`**
```python
def __init__(
    self,
    model_path: str = "/efs/shared/models/Llama-3.2-11B-Vision",  # Updated default
    device: str = "auto",
    use_8bit: bool = True,  # Enable 8-bit quantization by default for 11B
    max_new_tokens: int = 512,  # Increase for better responses
):
```

### 2. Model Interface Changes

The 11B model uses the standard transformers interface instead of the custom `answer_question` method.

**Update `_generate_response` method in `llama_vision_extractor.py`:**

```python
def _generate_response(self, prompt: str, image: Image.Image) -> str:
    """Generate response from Llama-Vision model."""
    try:
        # Check if this is the official Llama-3.2-11B-Vision model
        if hasattr(self.model, 'chat') or 'llama' in str(type(self.model)).lower():
            # Official Llama-3.2-11B-Vision model
            self.logger.debug("Using official Llama-3.2-11B-Vision interface")
            
            # Create messages in the expected format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Use the chat method if available
            if hasattr(self.model, 'chat'):
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    messages=messages,
                    temperature=0.1,
                    max_new_tokens=self.max_new_tokens
                )
            else:
                # Fallback to manual processing
                text_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Process image and text together (model-specific implementation)
                inputs = self.processor(
                    images=image,
                    text=text_prompt,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the prompt from response
                if text_prompt in response:
                    response = response.replace(text_prompt, "").strip()
                    
        elif hasattr(self.model, 'answer_question'):
            # Fallback for custom Llamavision models (current 1B implementation)
            self.logger.debug("Using custom Llamavision interface")
            response = self.model.answer_question(
                image=image,
                question=prompt,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.1,
            )
        else:
            raise ValueError("Unsupported model interface")
            
        return response
        
    except Exception as e:
        self.logger.error(f"Error in _generate_response: {e}")
        return ""
```

### 3. Memory and Performance Optimizations

**Enable 8-bit quantization by default for 11B model:**

```python
def __init__(self, ...):
    # Configure quantization for 11B model
    quantization_config = None
    if use_8bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offload if needed
                llm_int8_skip_modules=["vision_tower", "mm_projector"]  # Skip vision components
            )
            self.logger.info("Using 8-bit quantization for 11B model")
        except ImportError:
            self.logger.warning("BitsAndBytesConfig not available")
```

**Update device mapping for larger model:**

```python
# Load model with proper device mapping for 11B
self.model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # Enable automatic device mapping for multi-GPU
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    trust_remote_code=True,
    local_files_only=True,
    low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
)
```

### 4. Enhanced Prompts for 11B Model

The 11B model can handle more sophisticated prompts. Update `_get_extraction_prompt`:

```python
def _get_extraction_prompt(self, field: Optional[str] = None) -> str:
    """Enhanced prompts for Llama-3.2-11B-Vision model."""
    
    # Enhanced field-specific prompts for 11B model
    field_prompts = {
        "store_name": """Analyze this receipt image and identify the store or business name. 
Look for the main business name, usually displayed prominently at the top of the receipt.
Return only the store name, no additional text.""",
        
        "date": """Look at this receipt and find the date of purchase. 
Convert any date format you find to YYYY-MM-DD format.
Common formats include MM/DD/YYYY, DD/MM/YYYY, or DD-MM-YYYY.
Return only the date in YYYY-MM-DD format.""",
        
        "total_amount": """Find the total amount paid on this receipt.
Look for words like "Total", "Amount Due", or "Balance".
Include the currency symbol if visible.
Return only the total amount.""",
    }
    
    # Enhanced full extraction prompt
    full_extraction_prompt = """You are an expert receipt analysis system. Analyze this receipt image and extract all relevant information with high accuracy.

Extract the following information and format as a valid JSON object:

1. **Store Name**: The business or merchant name (usually at the top)
2. **Date**: Purchase date in YYYY-MM-DD format
3. **Time**: Purchase time in HH:MM format (24-hour)
4. **Total Amount**: Final amount paid (include currency symbol)
5. **Payment Method**: How the payment was made (cash, card, etc.)
6. **Receipt ID**: Receipt number, transaction ID, or reference number
7. **Items**: All purchased items with names and prices
8. **Tax Information**: Tax amount and/or rate
9. **Discounts**: Any discounts or promotions applied

Return ONLY a valid JSON object in this exact format:
{
  "store_name": "exact store name from receipt",
  "date": "YYYY-MM-DD",
  "time": "HH:MM",
  "total_amount": "amount with currency symbol",
  "payment_method": "payment method used",
  "receipt_id": "receipt number or transaction ID",
  "items": [
    {"item_name": "item name", "quantity": number, "price": "price with currency"}
  ],
  "tax_info": "tax amount and/or percentage",
  "discounts": "discount information or null if none"
}

Important: Return ONLY the JSON object, no additional text or explanations."""
    
    if field is not None and field in field_prompts:
        return field_prompts[field]
    else:
        return full_extraction_prompt
```

### 5. Hardware Requirements

**Minimum Requirements for 11B Model:**
- **GPU Memory**: 24GB VRAM (with 8-bit quantization)
- **System RAM**: 32GB+ recommended
- **Storage**: 25GB+ for model files
- **CUDA**: Version 11.8+ or 12.x

**Optimal Requirements:**
- **GPU Memory**: 48GB+ VRAM (for 16-bit precision)
- **System RAM**: 64GB+
- **Multiple GPUs**: Supported with `device_map="auto"`

### 6. Environment Setup for 11B Model

**Update `environment.yml` for production deployment:**

```yaml
name: llama_vision_11b_env
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pytorch>=2.0.0
  - pytorch-cuda=11.8  # or 12.1 for newer systems
  - transformers>=4.37.0
  - accelerate>=0.25.0
  - bitsandbytes>=0.41.0  # Required for 8-bit quantization
  - pillow>=10.0.0
  - pandas>=2.0.0
  - numpy>=1.24.0
  - tqdm>=4.65.0
  - pyyaml>=6.0
  - pip
  - pip:
    - typer[all]>=0.9.0
    - rich>=13.0.0

variables:
  KMP_DUPLICATE_LIB_OK: "TRUE"
  CUDA_VISIBLE_DEVICES: "0,1"  # Use multiple GPUs if available
  TRANSFORMERS_CACHE: "/efs/shared/cache"  # Shared cache location
```

### 7. Script Updates

**Update all script default paths:**

```bash
# scripts/test_llama_vision_extractor.py
python scripts/test_llama_vision_extractor.py \
  --model-path /efs/shared/models/Llama-3.2-11B-Vision \
  --use-8bit \
  --device auto \
  receipt.jpg

# scripts/batch_extract_receipts.py  
python scripts/batch_extract_receipts.py \
  --model-path /efs/shared/models/Llama-3.2-11B-Vision \
  --input-dir data/receipts \
  --output-dir results/batch_extraction \
  --use-8bit \
  --device auto
```

### 8. Model Download and Setup

**Download the official 11B model:**

```bash
# On the remote GPU host with internet access
huggingface-cli download meta-llama/Llama-3.2-11B-Vision \
  --local-dir /efs/shared/models/Llama-3.2-11B-Vision \
  --cache-dir /efs/shared/cache

# Verify model files
ls -la /efs/shared/models/Llama-3.2-11B-Vision/
```

### 9. Testing and Validation

**Test the 11B model setup:**

```bash
# 1. Test model loading
python -c "
from models.extractors.llama_vision_extractor import LlamaVisionExtractor
extractor = LlamaVisionExtractor(
    model_path='/efs/shared/models/Llama-3.2-11B-Vision',
    use_8bit=True
)
print('✅ 11B model loaded successfully')
"

# 2. Test single receipt extraction
python scripts/test_llama_vision_extractor.py \
  /path/to/test_receipt.jpg \
  --model-path /efs/shared/models/Llama-3.2-11B-Vision \
  --use-8bit \
  --verbose

# 3. Compare results with 1B model
python scripts/compare_models.py \
  --image test_receipt.jpg \
  --model-1b /Users/tod/PretrainedLLM/Llama-3.2-1B-Vision \
  --model-11b /efs/shared/models/Llama-3.2-11B-Vision
```

## Migration Strategy

### Phase 1: Preparation
1. **Download 11B model** to shared storage location
2. **Update configuration files** with new model paths
3. **Test model loading** without inference
4. **Verify hardware requirements** are met

### Phase 2: Code Adaptation  
1. **Update model interface** in `llama_vision_extractor.py`
2. **Enhance prompts** for better 11B model performance
3. **Enable 8-bit quantization** by default
4. **Update all script defaults** to use 11B model

### Phase 3: Testing and Validation
1. **Test single receipt extraction** with known examples
2. **Compare results** between 1B and 11B models
3. **Benchmark performance** and memory usage
4. **Validate extraction accuracy** on diverse receipts

### Phase 4: Production Deployment
1. **Update documentation** and README files
2. **Deploy to remote GPU hosts**
3. **Monitor performance** and memory usage
4. **Fine-tune configuration** based on results

## Expected Performance Improvements

With the 11B model, you should see:

- **Higher accuracy** in text extraction and understanding
- **Better handling** of complex receipt layouts
- **More consistent** JSON formatting and structure
- **Improved robustness** with varied receipt types
- **Better field detection** for edge cases

## Rollback Plan

If issues arise with the 11B model:

1. **Keep 1B model configuration** as backup
2. **Use environment variables** to switch models quickly:
   ```bash
   export LLAMA_MODEL_PATH="/Users/tod/PretrainedLLM/Llama-3.2-1B-Vision"  # Rollback
   export LLAMA_MODEL_PATH="/efs/shared/models/Llama-3.2-11B-Vision"       # Production
   ```
3. **Maintain separate conda environments** for each model version
4. **Use feature flags** in code to toggle between implementations

## Conclusion

This adaptation guide provides a comprehensive path to upgrade from the 1B development model to the production-ready 11B model. The key changes involve updating model interfaces, enhancing prompts, and ensuring adequate hardware resources for the larger model.

The 11B model will provide significantly better extraction accuracy and robustness for production receipt processing workloads.