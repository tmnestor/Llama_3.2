#!/usr/bin/env python3
"""Debug script to show exactly what images are being discovered."""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_vision.image.loaders import ImageLoader


def main():
    print("🔍 Debugging Image Discovery")
    print("=" * 50)
    
    loader = ImageLoader()
    discovered = loader.discover_images("datasets")
    
    print(f"Total categories found: {len(discovered)}")
    print()
    
    total_images = 0
    for category, images in discovered.items():
        count = len(images)
        total_images += count
        print(f"📁 {category}: {count} images")
        
        if count > 0:
            print(f"   Sample files: {[img.name for img in images[:3]]}")
            print(f"   Directory: {images[0].parent}")
        print()
    
    print(f"🔢 Total images: {total_images}")
    
    # Check for specific directories
    datasets_path = Path("datasets")
    if datasets_path.exists():
        print("\n📂 Contents of datasets directory:")
        for item in sorted(datasets_path.iterdir()):
            if item.is_dir():
                image_count = len(list(item.glob("*.png"))) + len(list(item.glob("*.jpg")))
                print(f"   📁 {item.name}/ ({image_count} images)")
            elif item.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                print(f"   🖼️  {item.name}")

if __name__ == "__main__":
    main()