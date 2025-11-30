import sys
import os
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("="*60)
print("DEBUGGING METADATA ERROR")
print("="*60)

try:
    from src.config import Config
    from src.input_processor.input_processor import InputProcessor
    
    Config.CLIP_MODEL_NAME = "ViT-B-32"
    Config.CLIP_PRETRAINED = "laion2b_s34b_b79k"
    
    input_path = "C:/battle-of-heuristics/tag-aware-smart-photo-album-generator/testing-images"
    print(f"\nTesting with: {input_path}")
    print(f"Path exists: {os.path.exists(input_path)}")
    
    print("\n1. Initializing InputProcessor...")
    processor = InputProcessor(input_path)
    print(f"   Found {len(processor.image_files)} images")
    
    print("\n2. Processing first image metadata...")
    for batch in processor.process_images(batch_size=1):
        print(f"   Batch size: {len(batch)}")
        if batch:
            first_img_data = batch[0]
            print(f"   Metadata keys: {list(first_img_data.keys())}")
            print(f"   Full metadata:")
            for key, value in first_img_data.items():
                print(f"     {key}: {value}")
        break
    
    print("\n3. Testing CLIP extractor...")
    from src.feature_extractor.clip_extractor import CLIPExtractor
    extractor = CLIPExtractor()
    
    # Load one image
    if processor.image_files:
        test_file = str(processor.image_files[0])
        print(f"   Loading: {test_file}")
        loaded = processor.load_and_preprocess_batch([test_file])
        if loaded:
            fp, img = loaded[0]
            print(f"   Image loaded: {img.size}")
            embeddings, tags = extractor.process_batch([img])
            print(f"   Embedding shape: {embeddings[0].shape}")
            print(f"   Tags: {tags[0][:5]}...")  # First 5 tags
    
    print("\n✅ Test successful!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
