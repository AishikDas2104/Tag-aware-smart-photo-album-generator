import numpy as np
from typing import Set, Dict, Any
from ..config import Config
from ..slide_builder.slide_constructor import Slide

class Scorer:
    """
    Computes transition scores between slides.
    """
    
    @staticmethod
    def calculate_transition_score(slide_a: Slide, slide_b: Slide) -> float:
        """
        Computes the transition score from slide A to slide B.
        Higher is better.
        """
        # 1. Tag Continuity (Intersection)
        # We want some shared context, but not necessarily identical tags
        common_tags = len(slide_a.tags & slide_b.tags)
        # Normalize somewhat (e.g., 5 common tags is "perfect")
        tag_score = min(common_tags / 5.0, 1.0)
        
        # 2. Visual Flow (Embedding Similarity)
        emb_a = slide_a.embedding
        emb_b = slide_b.embedding
        
        # Ensure flat
        if len(emb_a.shape) > 1: emb_a = emb_a.flatten()
        if len(emb_b.shape) > 1: emb_b = emb_b.flatten()
        
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        
        if norm_a > 0 and norm_b > 0:
            visual_score = np.dot(emb_a, emb_b) / (norm_a * norm_b)
        else:
            visual_score = 0.0
            
        # 3. Temporal Proximity (if available)
        time_score = 0.0
        ts_a = slide_a.metadata.get("timestamp")
        ts_b = slide_b.metadata.get("timestamp")
        
        if ts_a and ts_b:
            # Difference in hours
            diff_hours = abs((ts_a - ts_b).total_seconds()) / 3600.0
            
            # Exponential decay: 1.0 if same time, 0.5 after 4 hours, near 0 after 24h
            # We want to encourage temporal locality but allow jumps
            time_score = np.exp(-diff_hours / 4.0)
            
        # Weighted Combination
        total_score = (
            Config.W_TAGS * tag_score +
            Config.W_EMBEDDING * visual_score +
            Config.W_TEMPORAL * time_score
        )
        
        return float(total_score)
