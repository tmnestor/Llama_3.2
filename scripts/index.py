#!/usr/bin/env python3
"""
Script Index - List and describe all available utility scripts.

This script provides an overview of all utility scripts in the scripts/ directory.
"""

from pathlib import Path
from typing import Dict, List

SCRIPT_DESCRIPTIONS = {
    # Deployment scripts
    "deployment/v100_deployment_test.py": "Test V100 16GB deployment with different quantization options",
    "deployment/quantization/v100_quantization_analysis.py": "Analyze memory requirements for different quantization levels",
    "deployment/quantization/v100_quantization_fix.py": "Fix quantization issues for V100 deployment",
    # Setup scripts
    "setup/verify_setup.py": "Verify conda environment and package installation",
    # Debugging scripts
    "debugging/debug_llama_vision.py": "Debug Llama Vision model loading and inference",
    "debugging/debug_vision_model.py": "Debug vision model components",
    "debugging/model_checks/check_model_vision.py": "Check model vision capabilities",
    "debugging/fix_tokenizer.py": "Fix tokenizer configuration issues",
    # Testing scripts
    "testing/test_cpu_offload.py": "Test CPU offloading for large models",
    "testing/test_tensor_fix.py": "Test tensor operation fixes",
    "testing/quick_tests/test_extraction_gpu.sh": "Quick GPU extraction test",
    "testing/quick_tests/quick_test_bank.sh": "Quick test for bank statement processing",
    "testing/quick_tests/quick_test_invoice.sh": "Quick test for invoice processing",
}


def get_script_category(script_path: str) -> str:
    """Get the category of a script based on its path."""
    parts = script_path.split("/")
    if len(parts) > 1:
        return parts[0].title()
    return "Other"


def list_scripts_by_category() -> Dict[str, List[str]]:
    """Organize scripts by category."""
    categories = {}

    for script, description in SCRIPT_DESCRIPTIONS.items():
        category = get_script_category(script)
        if category not in categories:
            categories[category] = []
        categories[category].append((script, description))

    return categories


def print_script_index():
    """Print a formatted index of all scripts."""
    print("\n📚 Llama Vision Utility Scripts Index\n")
    print("=" * 80)

    categories = list_scripts_by_category()

    for category in sorted(categories.keys()):
        print(f"\n🔹 {category} Scripts")
        print("-" * 40)

        for script, description in sorted(categories[category]):
            script_name = script.split("/")[-1]
            indent = "  " * (script.count("/") - 1)
            print(f"{indent}📄 {script_name}")
            print(f"{indent}   {description}")

    print("\n" + "=" * 80)
    print("\n💡 Usage Tips:")
    print("   - Run scripts from the project root directory")
    print("   - Ensure conda environment is activated: conda activate vision_env")
    print(
        "   - Check script docstrings for detailed usage: python scripts/<path>/script.py --help"
    )
    print("")


def check_script_health():
    """Check if all listed scripts exist."""
    print("\n🔍 Checking script health...")

    missing_scripts = []
    scripts_dir = Path(__file__).parent

    for script in SCRIPT_DESCRIPTIONS:
        script_path = scripts_dir / script
        if not script_path.exists():
            missing_scripts.append(script)

    if missing_scripts:
        print("\n⚠️  Missing scripts:")
        for script in missing_scripts:
            print(f"   - {script}")
    else:
        print("✅ All scripts found!")


def suggest_cleanup():
    """Suggest cleanup actions for legacy scripts."""
    print("\n🧹 Cleanup Suggestions:")
    print("   1. Review scripts in legacy/ directory for removal")
    print("   2. Update old shell scripts to Python for consistency")
    print("   3. Add proper CLI interfaces using typer to utility scripts")
    print("   4. Create unit tests for critical utility functions")


if __name__ == "__main__":
    print_script_index()
    check_script_health()
    suggest_cleanup()
