#!/usr/bin/env python3
"""Script to rename all images in datasets directory to generic names."""

import shutil
from pathlib import Path


def rename_images_in_directory(directory_path: Path, create_mapping: bool = True):
    """Rename all images in a directory to generic names (image1.png, image2.png, etc.).
    
    Args:
        directory_path: Path to the directory containing images
        create_mapping: Whether to create a mapping file of old->new names
    """
    print(f"Processing directory: {directory_path}")
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    # Sort by name for consistent ordering
    image_files.sort(key=lambda x: x.name)
    
    print(f"Found {len(image_files)} images to rename")
    
    # Create mapping file if requested
    mapping = {}
    if create_mapping:
        mapping_file = directory_path / "image_name_mapping.txt"
        print(f"Creating mapping file: {mapping_file}")
    
    # Create temporary directory for renaming
    temp_dir = directory_path / "temp_rename"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Move all files to temp directory with new names
        for i, old_path in enumerate(image_files, 1):
            new_name = f"image{i}.png"
            temp_path = temp_dir / new_name
            
            # Copy file to temp directory with new name
            shutil.copy2(old_path, temp_path)
            
            # Store mapping
            mapping[old_path.name] = new_name
            
            print(f"  {old_path.name} -> {new_name}")
        
        # Step 2: Remove original files
        for old_path in image_files:
            old_path.unlink()
            print(f"  Removed: {old_path.name}")
        
        # Step 3: Move renamed files back to original directory
        for temp_file in temp_dir.iterdir():
            if temp_file.is_file():
                final_path = directory_path / temp_file.name
                shutil.move(temp_file, final_path)
                print(f"  Moved: {temp_file.name} to final location")
        
        # Step 4: Remove temp directory
        temp_dir.rmdir()
        
        # Step 5: Create mapping file
        if create_mapping and mapping:
            with mapping_file.open('w') as f:
                f.write("# Image name mapping (old_name -> new_name)\n")
                f.write("# Created by rename_images.py\n\n")
                for old_name, new_name in mapping.items():
                    f.write(f"{old_name} -> {new_name}\n")
            
            print(f"Mapping file created: {mapping_file}")
        
        print(f"Successfully renamed {len(image_files)} images")
        return mapping
        
    except Exception as e:
        print(f"Error during renaming: {e}")
        # Clean up temp directory if it exists
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def main():
    """Main function to rename images in datasets directory."""
    # Get the datasets directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    datasets_dir = project_root / "datasets"
    
    if not datasets_dir.exists():
        print(f"Error: datasets directory not found at {datasets_dir}")
        return
    
    print("Image Renaming Script")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Datasets directory: {datasets_dir}")
    
    # Rename images in main datasets directory
    print("\n1. Renaming images in main datasets directory...")
    rename_images_in_directory(datasets_dir, create_mapping=True)
    
    # Check for subdirectories and rename images there too
    subdirs = ['examples', 'synthetic_receipts', 'synthetic_bank_statements']
    
    for subdir in subdirs:
        subdir_path = datasets_dir / subdir
        if subdir_path.exists() and subdir_path.is_dir():
            print(f"\n2. Renaming images in {subdir} subdirectory...")
            rename_images_in_directory(subdir_path, create_mapping=True)
    
    # Check for synthetic_receipts/images
    synthetic_images_dir = datasets_dir / "synthetic_receipts" / "images"
    if synthetic_images_dir.exists() and synthetic_images_dir.is_dir():
        print("\n3. Renaming images in synthetic_receipts/images...")
        rename_images_in_directory(synthetic_images_dir, create_mapping=True)
    
    print("\n" + "=" * 50)
    print("Image renaming completed!")
    print("All images now have generic names (image1.png, image2.png, etc.)")
    print("Mapping files created showing old->new name relationships")


if __name__ == "__main__":
    main()