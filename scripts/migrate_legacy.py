#!/usr/bin/env python3
"""
Migrate legacy scripts to organized structure.

This script helps migrate old scripts to the new organized structure.
"""

import shutil
from pathlib import Path
from typing import Dict, List

# Mapping of old files to their new locations
MIGRATION_MAP = {
    # Legacy configuration files
    "prompt_config.py": "scripts/legacy/archive/prompt_config.py",
    "manifest.yaml": "scripts/legacy/archive/manifest.yaml",
    # Any remaining test files in root
    "test_*.py": "scripts/testing/",
    "debug_*.py": "scripts/debugging/",
    "check_*.py": "scripts/debugging/model_checks/",
    "fix_*.py": "scripts/debugging/",
    # Documentation that might be outdated
    "llama_vision_v100_implementation.md": "docs/implementation/v100_implementation.md",
    "LLAMA_VISION_TROUBLESHOOTING.md": "docs/troubleshooting/",
    "LLAMA_VISION_PACKAGE_IMPLEMENTATION.md": "docs/implementation/",
}


def find_legacy_files() -> List[Path]:
    """Find files that should be migrated."""
    project_root = Path(__file__).parent.parent
    legacy_files = []

    # Check for specific files
    for pattern in ["test_*.py", "debug_*.py", "check_*.py", "fix_*.py", "quick_*.sh"]:
        for file in project_root.glob(pattern):
            if file.parent == project_root:  # Only files in root
                legacy_files.append(file)

    # Check for other known legacy files
    for file in ["prompt_config.py", "manifest.yaml"]:
        file_path = project_root / file
        if file_path.exists():
            legacy_files.append(file_path)

    return legacy_files


def migrate_files(dry_run: bool = True) -> Dict[str, List[str]]:
    """Migrate legacy files to new structure."""
    project_root = Path(__file__).parent.parent
    migrations = {
        "moved": [],
        "archived": [],
        "skipped": [],
    }

    legacy_files = find_legacy_files()

    for file in legacy_files:
        # Determine destination
        if file.name.startswith("test_"):
            dest = project_root / "scripts" / "testing" / file.name
        elif file.name.startswith("debug_"):
            dest = project_root / "scripts" / "debugging" / file.name
        elif file.name.startswith("check_"):
            dest = project_root / "scripts" / "debugging" / "model_checks" / file.name
        elif file.name.startswith("fix_"):
            dest = project_root / "scripts" / "debugging" / file.name
        elif file.name.endswith(".sh"):
            dest = project_root / "scripts" / "testing" / "quick_tests" / file.name
        else:
            dest = project_root / "scripts" / "legacy" / "archive" / file.name

        # Check if destination exists
        if dest.exists():
            migrations["skipped"].append(f"{file.name} (already exists at destination)")
            continue

        # Perform migration
        if dry_run:
            action = "Would move" if "legacy" not in str(dest) else "Would archive"
            print(
                f"{action}: {file.relative_to(project_root)} → {dest.relative_to(project_root)}"
            )
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), str(dest))

        if "legacy" in str(dest):
            migrations["archived"].append(file.name)
        else:
            migrations["moved"].append(file.name)

    return migrations


def create_legacy_readme():
    """Create README for legacy directory."""
    readme_content = """# Legacy Scripts Archive

This directory contains scripts that are no longer actively used but are kept for reference.

## Archive Policy

Scripts are moved here when:
- They haven't been used in over 30 days
- They've been replaced by better implementations
- They were temporary fixes that are no longer needed
- They're from the old notebook-based architecture

## Before Deletion

Before permanently deleting scripts from this archive:
1. Ensure functionality is covered elsewhere
2. Check if any documentation references them
3. Verify no active branches use them
4. Wait at least 90 days after archiving

## Archived Scripts

See individual script files for their original purpose and why they were archived.
"""

    readme_path = Path(__file__).parent / "archive" / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(readme_content)


def main():
    """Run the migration process."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate legacy scripts")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files (default is dry run)",
    )

    args = parser.parse_args()

    print("\n🔄 Legacy Script Migration\n")
    print("=" * 60)

    # Find legacy files
    legacy_files = find_legacy_files()

    if not legacy_files:
        print("✅ No legacy files found in root directory!")
        return

    print(f"Found {len(legacy_files)} legacy files:\n")
    for file in legacy_files:
        print(f"  - {file.name}")

    print("\n" + "-" * 60 + "\n")

    # Perform migration
    if args.execute:
        print("Executing migration...\n")
        create_legacy_readme()
    else:
        print("DRY RUN - No files will be moved\n")

    migrations = migrate_files(dry_run=not args.execute)

    # Summary
    print("\n" + "=" * 60)
    print("\n📊 Migration Summary:")
    print(f"   Moved: {len(migrations['moved'])} files")
    print(f"   Archived: {len(migrations['archived'])} files")
    print(f"   Skipped: {len(migrations['skipped'])} files")

    if not args.execute:
        print("\n💡 To execute the migration, run:")
        print("   python scripts/migrate_legacy.py --execute")


if __name__ == "__main__":
    main()
