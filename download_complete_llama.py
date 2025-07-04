#!/usr/bin/env python3
"""
Download complete Llama-3.2-11B-Vision repository using Python.

This script downloads all files in the repository as flat files to $HOME/nfs_share/models
for offline access, with progress tracking and resume capability.
"""

import os
from pathlib import Path

from huggingface_hub import snapshot_download


def download_complete_llama():
    """Download the complete Llama-3.2-11B-Vision repository."""

    # Set token (ensure HF_TOKEN environment variable is set)
    if "HF_TOKEN" not in os.environ:
        raise ValueError("HF_TOKEN environment variable must be set")

    # Set up destination directory
    home_dir = Path.home()
    model_dir = home_dir / "nfs_share" / "models" / "Llama-3.2-11B-Vision"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Downloading complete Llama-3.2-11B-Vision repository")
    print(f"📁 Destination: {model_dir}")
    print("⏱️  This will download ~22GB of files...")
    print("")

    try:
        # Download complete repository as flat files
        snapshot_download(
            repo_id="meta-llama/Llama-3.2-11B-Vision",
            local_dir=str(model_dir),
            resume_download=True,  # Resume if interrupted
            local_files_only=False,  # Download from remote
            # Skip .git files and save as flat files (no HF cache format)
            ignore_patterns=["*.git*", "*.gitattributes"],
            cache_dir=None,  # Disable HuggingFace cache format
        )

        print("\n✅ Download complete!")
        print(f"📁 Model location: {model_dir}")
        print("🎯 Ready for offline access!")

        # List downloaded files
        if model_dir.exists():
            print("\n📋 Downloaded files:")
            for file in sorted(model_dir.rglob("*")):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"  {file.name}: {size_mb:.1f} MB")

        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("💡 The download can be resumed by running this script again")
        return False


if __name__ == "__main__":
    success = download_complete_llama()
    if success:
        print("\n🎉 SUCCESS! Llama-3.2-11B-Vision is ready for offline access!")
    else:
        print("\n💔 Download incomplete - try running again to resume")
