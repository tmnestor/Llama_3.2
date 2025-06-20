import os

from huggingface_hub import hf_hub_download


# Set HF_TOKEN environment variable before running this script
# export HF_TOKEN="your_token_here"
if "HF_TOKEN" not in os.environ:
    raise ValueError("HF_TOKEN environment variable must be set")
print("Downloading model-00003-of-00005.safetensors...")

hf_hub_download(
    repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
    filename="model-00003-of-00005.safetensors",
    local_dir="/Users/tod/PretrainedLLM/Llama-3.2-11B",
    local_dir_use_symlinks=False,
)
