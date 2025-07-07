#!/usr/bin/env python3
"""
Identify legacy scripts that should be migrated or removed.

This script helps identify old utility scripts and notebooks that
may need to be updated or removed.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def identify_legacy_patterns() -> Dict[str, List[str]]:
    """Identify files that match legacy patterns."""

    project_root = Path(__file__).parent.parent.parent
    legacy_patterns = {
        "Old test scripts": [],
        "Debugging scripts": [],
        "Temporary fixes": [],
        "Undocumented scripts": [],
        "Shell scripts to migrate": [],
    }

    # Check root directory for stray scripts
    for file in project_root.glob("*.py"):
        if file.name not in ["prompt_config.py", "setup.py"]:
            content = file.read_text()

            # Check for old patterns
            if (
                "if __name__ == '__main__':" in content
                and len(content.splitlines()) < 50
            ):
                legacy_patterns["Old test scripts"].append(
                    str(file.relative_to(project_root))
                )

            # Check for missing docstrings
            if not content.strip().startswith('"""') and not content.strip().startswith(
                "'''"
            ):
                legacy_patterns["Undocumented scripts"].append(
                    str(file.relative_to(project_root))
                )

            # Check for temporary fixes
            if any(
                word in file.name.lower() for word in ["fix", "temp", "old", "backup"]
            ):
                legacy_patterns["Temporary fixes"].append(
                    str(file.relative_to(project_root))
                )

    # Check for shell scripts
    for file in project_root.glob("*.sh"):
        legacy_patterns["Shell scripts to migrate"].append(
            str(file.relative_to(project_root))
        )

    return legacy_patterns


def analyze_script_usage() -> List[Tuple[str, datetime]]:
    """Analyze when scripts were last modified."""
    project_root = Path(__file__).parent.parent.parent
    script_ages = []

    for file in project_root.glob("*.py"):
        if file.name.startswith(("test_", "debug_", "check_", "fix_")):
            stat = file.stat()
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            script_ages.append((str(file.relative_to(project_root)), mod_time))

    return sorted(script_ages, key=lambda x: x[1])


def generate_migration_plan() -> Dict[str, str]:
    """Generate a migration plan for legacy scripts."""

    migration_plan = {
        # Legacy script -> New location
        "prompt_config.py": "llama_vision/config/legacy_prompts.py",
        "manifest.yaml": "config/legacy/manifest.yaml",
        "python -m tax_invoice_ner.cli extract au": "scripts/legacy/archive/",
    }

    return migration_plan


def print_legacy_analysis():
    """Print analysis of legacy scripts."""
    print("\n🔍 Legacy Script Analysis\n")
    print("=" * 80)

    # Identify legacy patterns
    legacy_patterns = identify_legacy_patterns()

    for category, files in legacy_patterns.items():
        if files:
            print(f"\n📁 {category}:")
            for file in files:
                print(f"   - {file}")

    # Show script ages
    print("\n⏰ Script Age Analysis:")
    script_ages = analyze_script_usage()

    if script_ages:
        print("\nOldest scripts (consider archiving):")
        for script, mod_time in script_ages[:5]:
            age_days = (datetime.now() - mod_time).days
            print(f"   - {script}: {age_days} days old")

    # Migration plan
    print("\n📋 Suggested Migration Plan:")
    migration_plan = generate_migration_plan()

    for old_path, new_path in migration_plan.items():
        print(f"   {old_path} → {new_path}")

    print("\n" + "=" * 80)
    print("\n💡 Recommendations:")
    print("   1. Archive scripts older than 30 days that aren't actively used")
    print("   2. Convert shell scripts to Python for consistency")
    print("   3. Add proper documentation to undocumented scripts")
    print("   4. Remove temporary fix scripts after verifying fixes are integrated")
    print("   5. Move all utility scripts to the scripts/ directory structure")


if __name__ == "__main__":
    print_legacy_analysis()
