#!/usr/bin/env python3
"""
Verify Llama-3.2-Vision package setup.

This script checks that all required dependencies are installed
and the package can be imported correctly.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(
            f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.11+)"
        )
        return False


def check_dependencies():
    """Check required dependencies."""
    print("\n📦 Checking dependencies...")

    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("typer", "Typer"),
        ("rich", "Rich"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
    ]

    all_good = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} (missing)")
            all_good = False

    return all_good


def check_torch_cuda():
    """Check PyTorch CUDA availability."""
    print("\n🚀 Checking PyTorch CUDA...")
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = (
                torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            )
            print(f"   ✅ CUDA available: {device_count} device(s)")
            print(f"   ✅ Primary device: {device_name}")
            return True
        else:
            print("   ⚠️  CUDA not available (CPU mode only)")
            return True  # Still valid, just slower
    except Exception as e:
        print(f"   ❌ Error checking CUDA: {e}")
        return False


def check_package_import():
    """Check if llama_vision package can be imported."""
    print("\n📦 Checking llama_vision package...")
    try:
        from llama_vision.config import load_config

        assert load_config is not None
        print("   ✅ llama_vision.config")

        from llama_vision.model import LlamaModelLoader

        assert LlamaModelLoader is not None
        print("   ✅ llama_vision.model")

        from llama_vision.extraction import TaxAuthorityParser

        assert TaxAuthorityParser is not None
        print("   ✅ llama_vision.extraction")

        from llama_vision.cli import llama_single

        assert llama_single is not None
        print("   ✅ llama_vision.cli")

        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False


def check_environment_files():
    """Check required configuration files."""
    print("\n📁 Checking configuration files...")

    files = [
        ("vision_env.yml", "Conda environment"),
        ("prompts.yaml", "Prompt configuration"),
        (".env", "Environment variables (optional)"),
    ]

    all_good = True
    for filename, description in files:
        filepath = Path(filename)
        if filepath.exists():
            print(f"   ✅ {description}: {filename}")
        else:
            if filename == ".env":
                print(f"   ⚠️  {description}: {filename} (optional)")
            else:
                print(f"   ❌ {description}: {filename} (missing)")
                all_good = False

    return all_good


def main():
    """Run all verification checks."""
    print("🔍 Llama-3.2-Vision Package Setup Verification\n")

    checks = [
        check_python_version(),
        check_dependencies(),
        check_torch_cuda(),
        check_environment_files(),
        check_package_import(),
    ]

    print("\n" + "=" * 50)

    if all(checks):
        print("🎉 All checks passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Ensure your model is available at the path in .env file")
        print("2. Try: python -m llama_vision.cli.llama_single --help")
        return 0
    else:
        print("❌ Some checks failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure conda environment is activated: conda activate vision_env")
        print("2. Install missing dependencies: conda env create -f vision_env.yml")
        return 1


if __name__ == "__main__":
    sys.exit(main())
