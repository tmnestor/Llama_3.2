"""Image loading utilities for Llama-3.2-Vision package."""

import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional

from ..utils import setup_logging


class ImageLoader:
    """Load and validate images for processing."""

    def __init__(
        self,
        log_level: str = "INFO",
        max_workers: int = 4,
        enable_parallel: bool = True,
    ):
        """Initialize image loader.

        Args:
            log_level: Logging level
            max_workers: Maximum number of worker threads for parallel processing
            enable_parallel: Whether to enable parallel processing
        """
        self.logger = setup_logging(log_level)
        self.supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        self.max_workers = max_workers
        self.enable_parallel = enable_parallel

    def discover_images(
        self, path: str, recursive: bool = True
    ) -> Dict[str, List[Path]]:
        """Discover images following InternVL pattern.

        Args:
            path: Base path to search for images
            recursive: Whether to search subdirectories

        Returns:
            Dictionary mapping category names to lists of image paths
        """
        base_path = Path(path)

        if not base_path.exists():
            self.logger.warning(f"Path does not exist: {base_path}")
            return {}

        self.logger.info(f"Discovering images in: {base_path}")

        # Get parent directory to find related data folders
        data_parent = base_path.parent if base_path.is_file() else base_path

        # Define image collections to discover - datasets directory only
        image_collections = {
            "configured_images": self._find_images_in_path(base_path, recursive),
        }
        
        # Check for datasets directory structure
        datasets_path = None
        if base_path.name == "datasets":
            datasets_path = base_path
        elif (base_path / "datasets").exists():
            datasets_path = base_path / "datasets"
        elif (data_parent / "datasets").exists():
            datasets_path = data_parent / "datasets"
        
        if datasets_path and datasets_path.exists():
            # All images are now in main datasets directory with generic names
            # Search only the main directory to avoid duplication
            image_collections["datasets"] = self._find_images_in_path(datasets_path, recursive=False)
            
            # Only search subdirectories if main directory has no images
            main_images = self._find_images_in_path(datasets_path, recursive=False)
            if not main_images:
                self.logger.info("No images found in main datasets directory, checking subdirectories...")
                # Check for specific subdirectories within datasets
                subdirs = ["examples", "synthetic_receipts", "synthetic_bank_statements", "test_images"]
                for subdir in subdirs:
                    subdir_path = datasets_path / subdir
                    if subdir_path.exists():
                        image_collections[f"datasets_{subdir}"] = self._find_images_in_path(subdir_path, recursive)
                        
                # Check for images subdirectory in synthetic_receipts
                synthetic_receipts_images = datasets_path / "synthetic_receipts" / "images"
                if synthetic_receipts_images.exists():
                    image_collections["synthetic_receipts_images"] = self._find_images_in_path(synthetic_receipts_images, recursive)
        else:
            self.logger.warning(f"No 'datasets' directory found in {base_path} or {data_parent}")
            self.logger.info("Please ensure images are organized in a 'datasets' directory")

        # Look for specific test files in datasets directory
        test_files = []
        
        if datasets_path:
            test_files.extend([
                datasets_path / "test_receipt.png",
                datasets_path / "test_receipt.jpg",
                datasets_path / "sample_receipt.png",
                datasets_path / "sample_receipt.jpg",
                datasets_path / "examples" / "test_receipt.png",
                datasets_path / "examples" / "test_receipt.jpg",
                datasets_path / "examples" / "sample_receipt.png",
                datasets_path / "examples" / "sample_receipt.jpg",
            ])

        test_receipt_images = [f for f in test_files if f.exists()]
        if test_receipt_images:
            image_collections["test_receipt"] = test_receipt_images

        # Filter existing files and log results
        available_images = {}
        total_images = 0

        for category, paths in image_collections.items():
            available_images[category] = [p for p in paths if p.exists()]
            count = len(available_images[category])
            total_images += count

            if count > 0:
                self.logger.info(f"Found {count} images in {category}")
                # Show sample filenames
                sample_names = [img.name for img in available_images[category][:3]]
                self.logger.debug(f"  Samples: {', '.join(sample_names)}")

        self.logger.info(f"Total images discovered: {total_images}")
        return available_images

    def _find_images_in_path(self, path: Path, recursive: bool = True) -> List[Path]:
        """Find all image files in a given path.

        Args:
            path: Path to search
            recursive: Whether to search recursively

        Returns:
            List of image file paths
        """
        if not path.exists():
            return []

        if self.enable_parallel and len(self.supported_extensions) > 1:
            return self._find_images_parallel(path, recursive)
        else:
            return self._find_images_sequential(path, recursive)

    def _find_images_sequential(self, path: Path, recursive: bool = True) -> List[Path]:
        """Find image files sequentially (original implementation)."""
        image_files = []

        if recursive:
            # Search recursively
            for ext in self.supported_extensions:
                image_files.extend(path.rglob(f"*{ext}"))
                image_files.extend(path.rglob(f"*{ext.upper()}"))
        else:
            # Search only in current directory
            for ext in self.supported_extensions:
                image_files.extend(path.glob(f"*{ext}"))
                image_files.extend(path.glob(f"*{ext.upper()}"))

        # Sort by modification time (newest first)
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return image_files

    def _find_images_parallel(self, path: Path, recursive: bool = True) -> List[Path]:
        """Find image files using parallel workers for better performance."""

        def search_extension(ext):
            """Search for files with a specific extension."""
            files = []
            if recursive:
                files.extend(path.rglob(f"*{ext}"))
                files.extend(path.rglob(f"*{ext.upper()}"))
            else:
                files.extend(path.glob(f"*{ext}"))
                files.extend(path.glob(f"*{ext.upper()}"))
            return files

        image_files = []

        # Use ThreadPoolExecutor for parallel file system operations
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(self.supported_extensions))
        ) as executor:
            # Submit tasks for each extension
            future_to_ext = {
                executor.submit(search_extension, ext): ext
                for ext in self.supported_extensions
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_ext):
                try:
                    files = future.result()
                    image_files.extend(files)
                except Exception as e:
                    ext = future_to_ext[future]
                    self.logger.warning(f"Error searching for {ext} files: {e}")

        # Sort by modification time (newest first)
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return image_files

    def load_image_batch(
        self, image_paths: List[str], validate: bool = True
    ) -> Dict[str, any]:
        """Load a batch of images with validation.

        Args:
            image_paths: List of paths to image files
            validate: Whether to validate images before loading

        Returns:
            Dictionary with loading results
        """
        results = {
            "successful": [],
            "failed": [],
            "total_requested": len(image_paths),
            "validation_errors": [],
        }

        self.logger.info(
            f"Loading batch of {len(image_paths)} images (parallel: {self.enable_parallel})"
        )

        if self.enable_parallel and len(image_paths) > 1:
            return self._load_image_batch_parallel(image_paths, validate, results)
        else:
            return self._load_image_batch_sequential(image_paths, validate, results)

    def _load_image_batch_sequential(
        self, image_paths: List[str], validate: bool, results: Dict
    ) -> Dict[str, any]:
        """Load images sequentially (original implementation)."""
        for i, image_path in enumerate(image_paths, 1):
            self.logger.debug(
                f"Processing image {i}/{len(image_paths)}: {Path(image_path).name}"
            )

            try:
                if validate:
                    from .preprocessing import validate_image

                    is_valid, error_msg = validate_image(image_path)

                    if not is_valid:
                        results["validation_errors"].append(
                            {"path": image_path, "error": error_msg}
                        )
                        results["failed"].append(image_path)
                        continue

                # Image is valid, add to successful list
                results["successful"].append(image_path)

            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
                results["failed"].append(image_path)

        success_rate = len(results["successful"]) / len(image_paths) * 100
        self.logger.info(
            f"Batch loading completed: {len(results['successful'])}/{len(image_paths)} successful ({success_rate:.1f}%)"
        )

        return results

    def _load_image_batch_parallel(
        self, image_paths: List[str], validate: bool, results: Dict
    ) -> Dict[str, any]:
        """Load images using parallel workers for better performance."""

        def process_image(image_path: str):
            """Process a single image with validation."""
            try:
                if validate:
                    from .preprocessing import validate_image

                    is_valid, error_msg = validate_image(image_path)

                    if not is_valid:
                        return {
                            "status": "validation_failed",
                            "path": image_path,
                            "error": error_msg,
                        }

                # Image is valid
                return {"status": "success", "path": image_path}

            except Exception as e:
                return {"status": "error", "path": image_path, "error": str(e)}

        # Process images in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submit all image processing tasks
            future_to_path = {
                executor.submit(process_image, image_path): image_path
                for image_path in image_paths
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_path):
                try:
                    result = future.result()

                    if result["status"] == "success":
                        results["successful"].append(result["path"])
                    elif result["status"] == "validation_failed":
                        results["validation_errors"].append(
                            {"path": result["path"], "error": result["error"]}
                        )
                        results["failed"].append(result["path"])
                    else:  # error
                        self.logger.error(
                            f"Error processing {result['path']}: {result['error']}"
                        )
                        results["failed"].append(result["path"])

                except Exception as e:
                    image_path = future_to_path[future]
                    self.logger.error(f"Unexpected error processing {image_path}: {e}")
                    results["failed"].append(image_path)

        success_rate = len(results["successful"]) / len(image_paths) * 100
        self.logger.info(
            f"Parallel batch loading completed: {len(results['successful'])}/{len(image_paths)} successful ({success_rate:.1f}%)"
        )

        return results

    def get_image_categories(
        self, discovered_images: Dict[str, List[Path]]
    ) -> Dict[str, dict]:
        """Get detailed information about discovered image categories.

        Args:
            discovered_images: Result from discover_images()

        Returns:
            Dictionary with category information
        """
        categories = {}

        for category, images in discovered_images.items():
            if not images:
                continue

            # Calculate total size
            total_size = sum(img.stat().st_size for img in images if img.exists())

            # Get file format distribution
            formats = {}
            for img in images:
                ext = img.suffix.lower()
                formats[ext] = formats.get(ext, 0) + 1

            categories[category] = {
                "count": len(images),
                "total_size_mb": total_size / (1024 * 1024),
                "formats": formats,
                "sample_files": [img.name for img in images[:5]],
                "newest_file": max(images, key=lambda x: x.stat().st_mtime).name
                if images
                else None,
                "oldest_file": min(images, key=lambda x: x.stat().st_mtime).name
                if images
                else None,
            }

        return categories

    def filter_images_by_type(
        self, image_paths: List[Path], document_types: List[str]
    ) -> List[Path]:
        """Filter images based on likely document type from filename.

        Args:
            image_paths: List of image paths to filter
            document_types: List of document types to filter for (e.g., ['receipt', 'invoice'])

        Returns:
            Filtered list of image paths
        """
        filtered_images = []

        for image_path in image_paths:
            filename_lower = image_path.name.lower()

            # Check if filename contains any of the desired document types
            if any(doc_type.lower() in filename_lower for doc_type in document_types):
                filtered_images.append(image_path)

        self.logger.info(
            f"Filtered {len(filtered_images)} images from {len(image_paths)} based on types: {document_types}"
        )

        return filtered_images

    def create_image_manifest(
        self,
        discovered_images: Dict[str, List[Path]],
        output_path: Optional[str] = None,
    ) -> Dict[str, any]:
        """Create a manifest of all discovered images.

        Args:
            discovered_images: Result from discover_images()
            output_path: Optional path to save manifest JSON file

        Returns:
            Manifest dictionary
        """
        manifest = {
            "created_at": str(Path(__file__).stat().st_mtime),
            "total_categories": len(
                [cat for cat, imgs in discovered_images.items() if imgs]
            ),
            "total_images": sum(len(imgs) for imgs in discovered_images.values()),
            "categories": {},
        }

        for category, images in discovered_images.items():
            if not images:
                continue

            manifest["categories"][category] = {
                "count": len(images),
                "images": [
                    {
                        "name": img.name,
                        "path": str(img),
                        "size_bytes": img.stat().st_size,
                        "modified": img.stat().st_mtime,
                    }
                    for img in images
                ],
            }

        # Save manifest if output path provided
        if output_path:
            import json

            manifest_path = Path(output_path)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)

            with manifest_path.open("w") as f:
                json.dump(manifest, f, indent=2)

            self.logger.info(f"Image manifest saved to: {manifest_path}")

        return manifest
