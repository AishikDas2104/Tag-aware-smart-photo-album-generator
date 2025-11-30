import sys
import os
import traceback
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import Config
from src.input_processor.input_processor import InputProcessor
from src.feature_extractor.clip_extractor import CLIPExtractor
from src.slide_builder.slide_constructor import SlideConstructor
from src.indexing.tag_index import TagIndex
from src.indexing.embedding_index import EmbeddingIndex
from src.sequencing.sequence_optimizer import SequenceOptimizer
from src.output_generator.generator import OutputGenerator

def reproduce():
    print("=== Starting Reproduction Script ===")
    # Patch config to use smaller model for testing
    Config.CLIP_MODEL_NAME = "ViT-B-32"
    Config.CLIP_PRETRAINED = "laion2b_s34b_b79k"
    print(f"   Using model: {Config.CLIP_MODEL_NAME} ({Config.CLIP_PRETRAINED})")
    
    input_dir = "testing-images"
    output_dir = "tests/reproduce_output"
    
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} does not exist.")
        return

    try:
        print("1. Initializing InputProcessor...")
        processor = InputProcessor(input_dir)
        
        print(f"   Found {len(processor.image_files)} images.")
        if len(processor.image_files) == 0:
            print("   No images found. Exiting.")
            return

        print("2. Processing images (Metadata)...")
        all_metadata = []
        for batch in processor.process_images(batch_size=32):
            all_metadata.extend(batch)
        print(f"   Extracted metadata for {len(all_metadata)} images.")

        print("3. Initializing CLIPExtractor...")
        extractor = CLIPExtractor()
        
        print("4. Extracting Features...")
        processed_data = []
        batch_size = 32
        
        for i in range(0, len(all_metadata), batch_size):
            batch_meta = all_metadata[i:i+batch_size]
            batch_paths = [m["filepath"] for m in batch_meta]
            
            loaded = processor.load_and_preprocess_batch(batch_paths)
            if not loaded:
                continue
                
            imgs = [x[1] for x in loaded]
            
            embeddings, tags_batch = extractor.process_batch(imgs)
            
            for j, (fp, img) in enumerate(loaded):
                meta = batch_meta[j]
                meta["embedding"] = embeddings[j]
                meta["tags"] = tags_batch[j]
                processed_data.append(meta)
            print(f"   Processed batch {i//batch_size + 1}")

        print("5. Building Slides...")
        constructor = SlideConstructor()
        slides = constructor.build_slides(processed_data)
        print(f"   Created {len(slides)} slides.")

        print("6. Building Indexes...")
        tag_index = TagIndex()
        tag_index.build(slides)
        
        emb_index = EmbeddingIndex()
        emb_index.build(slides)

        print("7. Optimizing Sequence...")
        optimizer = SequenceOptimizer(tag_index, emb_index)
        ordered_slides = optimizer.optimize(slides)
        print(f"   Sequence length: {len(ordered_slides)}")

        print("8. Generating Output...")
        gen = OutputGenerator(output_dir)
        gen.save_json(ordered_slides)
        gen.save_csv(ordered_slides)
        
        if Config.VIDEO_ENABLED:
            print("   Generating video...")
            gen.generate_video(ordered_slides)

        print("=== Success! ===")

    except Exception:
        print("\n!!! EXCEPTION CAUGHT !!!")
        traceback.print_exc()

if __name__ == "__main__":
    reproduce()
