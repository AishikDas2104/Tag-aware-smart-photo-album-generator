import os
from pathlib import Path

class Config:
    # ==========================================
    # Paths
    # ==========================================
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # ==========================================
    # Image Processing
    # ==========================================
    IMAGE_SIZE = (512, 512)  # Standard size for CLIP
    BATCH_SIZE = 32
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
    
    # ==========================================
    # Feature Extraction
    # ==========================================
    CLIP_MODEL_NAME = "ViT-L-14"  # Default
    CLIP_PRETRAINED = "laion2b_s32b_b82k"  # Default
    
    MODEL_MAPPING = {
        "ViT-L-14": "laion2b_s32b_b82k",  # High Quality, Slow (1.7GB)
        "ViT-B-32": "laion2b_s34b_b79k",  # Faster, Good Quality (600MB)
    }
    
    TAG_THRESHOLD = 0.25
    
    # ==========================================
    # Pairing Logic
    # ==========================================
    PAIRING_STRATEGY = "hybrid"  # "tag", "embedding", "hybrid"
    
    # ==========================================
    # Sequencing Weights
    # ==========================================
    W_TAGS = 1.0        # Weight for common tags
    W_EMBEDDING = 2.0   # Weight for visual similarity
    W_TEMPORAL = 0.5    # Weight for time proximity
    
    # ==========================================
    # Refinement
    # ==========================================
    WINDOW_SIZE = 100
    MAX_REFINEMENT_ITERATIONS = 10
    
    # ==========================================
    # Output
    # ==========================================
    VIDEO_ENABLED = True
    SLIDE_DURATION = 3.0  # seconds
    TRANSITION_TYPE = "crossfade"
    TRANSITION_DURATION = 0.5
    
    @classmethod
    def ensure_dirs(cls):
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
