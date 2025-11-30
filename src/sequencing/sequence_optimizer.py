import random
import numpy as np
from typing import List, Set, Dict, Optional
from tqdm import tqdm
import logging

from ..config import Config
from ..slide_builder.slide_constructor import Slide
from ..indexing.tag_index import TagIndex
from ..indexing.embedding_index import EmbeddingIndex
from .scoring import Scorer

logger = logging.getLogger(__name__)

class SequenceOptimizer:
    """
    Optimizes the order of slides to create a coherent story.
    Uses a two-stage approach:
    1. Greedy construction with lookahead/candidates
    2. Local refinement (window-based optimization)
    """
    
    def __init__(self, tag_index: TagIndex, embedding_index: EmbeddingIndex):
        self.tag_index = tag_index
        self.embedding_index = embedding_index
        
    def optimize(self, slides: List[Slide]) -> List[Slide]:
        """
        Main entry point for sequence optimization.
        """
        if not slides:
            return []
            
        logger.info(f"Starting sequence optimization for {len(slides)} slides...")
        
        # Stage 1: Greedy Construction
        sequence = self._build_greedy_sequence(slides)
        
        # Stage 2: Local Refinement
        refined_sequence = self._refine_sequence(sequence)
        
        return refined_sequence

    def _build_greedy_sequence(self, slides: List[Slide]) -> List[Slide]:
        """
        Builds an initial sequence by greedily selecting the best next slide.
        """
        remaining = {s.id: s for s in slides}
        
        # Pick a starting slide (e.g., earliest timestamp)
        # Use a very old date as fallback for slides without timestamps
        from datetime import datetime
        fallback_date = datetime(1900, 1, 1)
        start_slide = min(slides, key=lambda s: s.metadata.get("timestamp") or fallback_date)
        
        sequence = [start_slide]
        del remaining[start_slide.id]
        
        pbar = tqdm(total=len(slides), desc="Building Sequence")
        pbar.update(1)
        
        while remaining:
            current = sequence[-1]
            
            # Get candidates
            candidates = self._get_candidates(current, remaining)
            
            # If no good candidates found, pick random fallback
            if not candidates:
                # Pick nearest in time if possible, else random
                # For efficiency, just pick random from remaining
                next_slide = random.choice(list(remaining.values()))
            else:
                # Score candidates
                best_score = -float('inf')
                next_slide = None
                
                for candidate in candidates:
                    score = Scorer.calculate_transition_score(current, candidate)
                    if score > best_score:
                        best_score = score
                        next_slide = candidate
                
                # Fallback if something went wrong
                if next_slide is None:
                    next_slide = random.choice(list(remaining.values()))
            
            sequence.append(next_slide)
            del remaining[next_slide.id]
            pbar.update(1)
            
        pbar.close()
        return sequence

    def _get_candidates(self, current: Slide, remaining: Dict[int, Slide], k: int = 50) -> List[Slide]:
        """
        Retrieves candidate slides using indexes.
        """
        candidates_map = {}
        
        # 1. Tag-based candidates
        # Look for slides sharing tags
        for tag in current.tags:
            matches = self.tag_index.get_slides_by_tag(tag)
            for match in matches:
                if match.id in remaining:
                    candidates_map[match.id] = match
                    if len(candidates_map) >= k:
                        break
            if len(candidates_map) >= k:
                break
                
        # 2. Embedding-based candidates
        # Find nearest neighbors
        neighbors = self.embedding_index.search(current.embedding, k=k)
        for slide_id, _ in neighbors:
            if slide_id in remaining:
                candidates_map[slide_id] = remaining[slide_id]
                
        # 3. Random fallback (exploration)
        # Add a few random slides to avoid getting stuck in local optima
        remaining_ids = list(remaining.keys())
        if len(remaining_ids) > 0:
            num_random = min(5, len(remaining_ids))
            random_ids = random.sample(remaining_ids, num_random)
            for rid in random_ids:
                candidates_map[rid] = remaining[rid]
                
        return list(candidates_map.values())

    def _refine_sequence(self, sequence: List[Slide]) -> List[Slide]:
        """
        Refines the sequence using local optimization (swaps).
        """
        logger.info("Refining sequence...")
        
        window_size = Config.WINDOW_SIZE
        improved = True
        iteration = 0
        
        while improved and iteration < Config.MAX_REFINEMENT_ITERATIONS:
            improved = False
            iteration += 1
            
            # Sliding window optimization
            # We can just iterate through the list and try swapping adjacent elements
            # Or use windows. Let's try simple adjacent swaps first.
            
            # Calculate initial total score (approximate)
            # total_score = sum(Scorer.calculate_transition_score(sequence[i], sequence[i+1]) for i in range(len(sequence)-1))
            
            swaps = 0
            for i in range(len(sequence) - 2):
                # Consider window of 3: A -> B -> C
                # Try swapping B and C: A -> C -> B
                # We need to check connections:
                # Original: (A->B) + (B->C) + (C->D)
                # Swapped:  (A->C) + (C->B) + (B->D)
                
                a = sequence[i]
                b = sequence[i+1]
                c = sequence[i+2]
                
                # Current score
                s1 = Scorer.calculate_transition_score(a, b)
                s2 = Scorer.calculate_transition_score(b, c)
                current_local_score = s1 + s2
                
                # Swapped score
                s1_new = Scorer.calculate_transition_score(a, c)
                s2_new = Scorer.calculate_transition_score(c, b)
                new_local_score = s1_new + s2_new
                
                # If we have a D, include it
                if i + 3 < len(sequence):
                    d = sequence[i+3]
                    s3 = Scorer.calculate_transition_score(c, d)
                    current_local_score += s3
                    
                    s3_new = Scorer.calculate_transition_score(b, d)
                    new_local_score += s3_new
                
                if new_local_score > current_local_score:
                    # Perform swap
                    sequence[i+1], sequence[i+2] = sequence[i+2], sequence[i+1]
                    improved = True
                    swaps += 1
            
            logger.info(f"Refinement iteration {iteration}: {swaps} swaps performed.")
            
        return sequence
