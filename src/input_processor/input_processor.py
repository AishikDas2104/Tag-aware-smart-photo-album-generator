import os
from pathlib import Path
from typing import Generator, List, Dict, Any, Tuple
from PIL import Image
from tqdm import tqdm
import logging

from ..config import Config
from .metadata_extractor import MetadataExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InputProcessor:
    """
    Handles loading, preprocessing, and metadata extraction for image collections.
    Designed to handle large datasets efficiently.
    """
    
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        self.supported_extensions = Config.SUPPORTED_EXTENSIONS
        self.image_files = self._scan_directory()
        logger.info(f"Found {len(self.image_files)} images in {self.input_dir}")

    def _scan_directory(self) -> List[Path]:
        """
        Recursively finds all supported image files in the input directory.
        """
        files = []
        for ext in self.supported_extensions:
            # Case insensitive search would be better, but glob is case sensitive on some OS
            # For Windows it's usually fine, but let's be explicit if needed.
            # Here we just look for lowercase extensions as defined in Config.
            files.extend(list(self.input_dir.rglob(f"*{ext}")))
            files.extend(list(self.input_dir.rglob(f"*{ext.upper()}")))
        return sorted(list(set(files)))  # Remove duplicates if any

    def process_images(self, batch_size: int = Config.BATCH_SIZE) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Yields batches of processed image metadata and loaded PIL images (optional).
        For the initial metadata pass, we might not want to keep images in memory.
        
        This function returns metadata dictionaries.
        """
        batch = []
        for file_path in tqdm(self.image_files, desc="Processing Images"):
            metadata = MetadataExtractor.extract(str(file_path))
            
            if "error" not in metadata:
                batch.append(metadata)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch

    def load_and_preprocess_batch(self, file_paths: List[str], target_size: Tuple[int, int] = Config.IMAGE_SIZE) -> List[Tuple[str, Image.Image]]:
        """
        Loads a specific batch of images and resizes them for model consumption.
        Returns list of (filepath, PIL_Image).
        """
        processed_images = []
        for fp in file_paths:
            try:
                with Image.open(fp) as img:
                    # Convert to RGB to handle RGBA/P modes
                    img = img.convert('RGB')
                    # Resize
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    processed_images.append((fp, img))
            except Exception as e:
                logger.error(f"Failed to load/preprocess {fp}: {e}")
        
        return processed_images

    def get_total_images(self) -> int:
        return len(self.image_files)

if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        processor = InputProcessor(sys.argv[1])
        for batch in processor.process_images(batch_size=5):
            print(f"Processed batch of {len(batch)} images")
            print(batch[0]) # Print first item of batch
            break
