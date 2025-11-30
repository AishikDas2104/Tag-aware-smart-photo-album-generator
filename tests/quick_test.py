import sys
import os
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

print("=== Quick Test ===", flush=True)

try:
    from src.config import Config
    from src.input_processor.input_processor import InputProcessor
    
    # Use ViT-B-32 like user selected
    Config.CLIP_MODEL_NAME = "ViT-B-32"
    Config.CLIP_PRETRAINED = "laion2b_s34b_b79k"
    
    input_dir = "testing-images"
    print(f"Testing with: {input_dir}", flush=True)
    
    processor = InputProcessor(input_dir)
    print(f"Found {len(processor.image_files)} images", flush=True)
    
    print("Processing first batch...", flush=True)
    for batch in processor.process_images(batch_size=2):
        print(f"Batch size: {len(batch)}", flush=True)
        for img_data in batch:
            print(f"  - {img_data['filepath']}", flush=True)
        break  # Just test first batch
    
    print("\nTesting CLIP...", flush=True)
    from src.feature_extractor.clip_extractor import CLIPExtractor
    
    extractor = CLIPExtractor()
    print("CLIP loaded successfully!", flush=True)
    
    # Load one image
    loaded = processor.load_and_preprocess_batch([str(processor.image_files[0])])
    if loaded:
        fp, img = loaded[0]
        print(f"Processing: {fp}", flush=True)
        embeddings, tags = extractor.process_batch([img])
        print(f"Tags: {tags[0]}", flush=True)
    
    print("\n=== Test Successful ===", flush=True)
    
except Exception:
    print("\n!!! ERROR !!!", flush=True)
    traceback.print_exc()
    sys.exit(1)
