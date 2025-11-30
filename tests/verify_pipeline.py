import sys
import os
import shutil
from pathlib import Path
import numpy as np
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Mock faiss if not installed
try:
    import faiss
except ImportError:
    import unittest.mock as mock
    import numpy as np
    
    faiss_mock = mock.Mock()
    # Configure search to return (scores, indices)
    # Shape: (1, k)
    def mock_search(q, k):
        return np.zeros((1, k), dtype=np.float32), np.zeros((1, k), dtype=np.int64)
        
    faiss_mock.IndexFlatIP.return_value.search.side_effect = mock_search
    faiss_mock.IndexIVFFlat.return_value.search.side_effect = mock_search
    
    sys.modules['faiss'] = faiss_mock
    print("Warning: faiss not found, using mock.")

from src.config import Config
from src.slide_builder.slide_constructor import SlideConstructor
from src.indexing.tag_index import TagIndex
from src.indexing.embedding_index import EmbeddingIndex
from src.sequencing.sequence_optimizer import SequenceOptimizer
from src.output_generator.generator import OutputGenerator

# Mock Data Generation
def generate_mock_data(num_images=20):
    data = []
    for i in range(num_images):
        # Alternate orientation
        orientation = "landscape" if i % 3 == 0 else "portrait"
        
        # Mock embedding (random)
        embedding = np.random.rand(768).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        # Mock tags
        tags = ["nature", "outdoor"]
        if i % 2 == 0: tags.append("beach")
        if i % 5 == 0: tags.append("person")
        
        data.append({
            "filepath": f"mock_img_{i}.jpg",
            "tags": tags,
            "embedding": embedding,
            "orientation": orientation,
            "metadata": {
                "timestamp": None,
                "gps_coords": None
            }
        })
    return data

def verify_pipeline():
    print("=== Starting Pipeline Verification ===")
    
    # 1. Generate Mock Data
    print("1. Generating mock data...")
    images_data = generate_mock_data(20)
    
    # 2. Build Slides
    print("2. Building slides...")
    constructor = SlideConstructor()
    slides = constructor.build_slides(images_data)
    print(f"   Created {len(slides)} slides from {len(images_data)} images.")
    
    # 3. Build Indexes
    print("3. Building indexes...")
    tag_index = TagIndex()
    tag_index.build(slides)
    
    emb_index = EmbeddingIndex()
    emb_index.build(slides)
    
    # 4. Optimize Sequence
    print("4. Optimizing sequence...")
    optimizer = SequenceOptimizer(tag_index, emb_index)
    ordered_slides = optimizer.optimize(slides)
    print(f"   Optimized sequence length: {len(ordered_slides)}")
    
    # 5. Generate Output
    print("5. Generating output...")
    output_dir = "tests/output"
    gen = OutputGenerator(output_dir)
    
    # Disable video for test
    Config.VIDEO_ENABLED = False
    
    json_path = gen.save_json(ordered_slides)
    csv_path = gen.save_csv(ordered_slides)
    
    print(f"   Saved JSON to {json_path}")
    print(f"   Saved CSV to {csv_path}")
    
    print("=== Verification Complete: SUCCESS ===")

if __name__ == "__main__":
    verify_pipeline()
