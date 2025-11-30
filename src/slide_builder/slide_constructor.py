from typing import List, Dict, Any, Set, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging
from ..config import Config

logger = logging.getLogger(__name__)

@dataclass
class Slide:
    id: int
    image_ids: List[str]  # List of filepaths or IDs
    tags: Set[str]
    embedding: np.ndarray
    orientation: str  # "landscape" or "paired_portrait"
    metadata: Dict[str, Any]

class SlideConstructor:
    """
    Constructs slides from processed images.
    Implements pairing logic for portrait images.
    """
    
    def __init__(self, pairing_strategy: str = Config.PAIRING_STRATEGY):
        self.pairing_strategy = pairing_strategy
        self.slide_counter = 0

    def build_slides(self, images_data: List[Dict[str, Any]]) -> List[Slide]:
        """
        Main entry point. Separates landscape and portrait images,
        pairs portraits, and returns list of all slides.
        
        images_data expected format:
        {
            "filepath": str,
            "tags": List[str],
            "embedding": np.ndarray (or torch.Tensor),
            "orientation": str,
            "metadata": dict
        }
        """
        landscapes = []
        portraits = []
        
        for img in images_data:
            if img["orientation"] == "landscape":
                landscapes.append(img)
            else:
                portraits.append(img)
                
        logger.info(f"Found {len(landscapes)} landscapes and {len(portraits)} portraits.")
        
        slides = []
        
        # 1. Process landscapes (Single slides)
        for img in landscapes:
            slides.append(self._create_single_slide(img))
            
        # 2. Process portraits (Pairing)
        paired_slides, leftovers = self._pair_portraits(portraits)
        slides.extend(paired_slides)
        
        # 3. Handle leftovers (Single portrait slides)
        for img in leftovers:
            slides.append(self._create_single_slide(img))
            
        logger.info(f"Created {len(slides)} total slides.")
        return slides

    def _create_single_slide(self, img_data: Dict[str, Any]) -> Slide:
        self.slide_counter += 1
        
        # Ensure embedding is numpy array
        emb = img_data["embedding"]
        if hasattr(emb, "numpy"):
            emb = emb.numpy()
        if len(emb.shape) > 1:
            emb = emb.flatten()
            
        return Slide(
            id=self.slide_counter,
            image_ids=[img_data["filepath"]],
            tags=set(img_data["tags"]),
            embedding=emb,
            orientation=img_data["orientation"],
            metadata=img_data["metadata"]
        )

    def _pair_portraits(self, portraits: List[Dict[str, Any]]) -> Tuple[List[Slide], List[Dict[str, Any]]]:
        """
        Pairs portrait images using the selected strategy.
        Returns (list_of_slides, list_of_unpaired_images).
        """
        if not portraits:
            return [], []
            
        pairs = []
        used_indices = set()
        
        # Sort by timestamp to pair temporally close images first?
        # Or just greedy matching? Let's do greedy matching for now.
        # For a more optimal approach, we could use max-weight matching (blossom algorithm),
        # but that's O(N^3). Greedy is O(N^2).
        
        # Optimization: Sort by timestamp to limit search window
        portraits.sort(key=lambda x: x["metadata"].get("timestamp") or 0)
        
        for i in range(len(portraits)):
            if i in used_indices:
                continue
                
            best_match_idx = -1
            best_score = -1.0
            
            # Search window: look ahead 50 images to find a match
            # This balances quality with performance
            search_end = min(len(portraits), i + 50)
            
            for j in range(i + 1, search_end):
                if j in used_indices:
                    continue
                    
                score = self._calculate_pairing_score(portraits[i], portraits[j])
                
                if score > best_score:
                    best_score = score
                    best_match_idx = j
            
            # Threshold for pairing?
            # For now, if we found any match, we take the best one.
            if best_match_idx != -1:
                pairs.append(self._create_paired_slide(portraits[i], portraits[best_match_idx]))
                used_indices.add(i)
                used_indices.add(best_match_idx)
                
        # Collect leftovers
        leftovers = [portraits[i] for i in range(len(portraits)) if i not in used_indices]
        
        return pairs, leftovers

    def _create_paired_slide(self, img1: Dict[str, Any], img2: Dict[str, Any]) -> Slide:
        self.slide_counter += 1
        
        # Combine tags (Union)
        combined_tags = set(img1["tags"]) | set(img2["tags"])
        
        # Combine embeddings (Average)
        emb1 = img1["embedding"]
        emb2 = img2["embedding"]
        if hasattr(emb1, "numpy"): emb1 = emb1.numpy()
        if hasattr(emb2, "numpy"): emb2 = emb2.numpy()
        
        avg_embedding = (emb1 + emb2) / 2.0
        if len(avg_embedding.shape) > 1:
            avg_embedding = avg_embedding.flatten()
            
        return Slide(
            id=self.slide_counter,
            image_ids=[img1["filepath"], img2["filepath"]],
            tags=combined_tags,
            embedding=avg_embedding,
            orientation="paired_portrait",
            metadata={
                "timestamp_start": min(img1["metadata"].get("timestamp") or 0, img2["metadata"].get("timestamp") or 0),
                "timestamp_end": max(img1["metadata"].get("timestamp") or 0, img2["metadata"].get("timestamp") or 0),
                "gps_1": img1["metadata"].get("gps_coords"),
                "gps_2": img2["metadata"].get("gps_coords")
            }
        )

    def _calculate_pairing_score(self, img1: Dict[str, Any], img2: Dict[str, Any]) -> float:
        """
        Calculates how well two images fit together in a slide.
        """
        # 1. Tag Diversity (Union size) - we want rich slides
        # But maybe we want shared tags for coherence?
        # Battle of Heuristics: "Maximize combined semantic richness"
        tags1 = set(img1["tags"])
        tags2 = set(img2["tags"])
        tag_score = len(tags1 | tags2)
        
        # 2. Visual Similarity (Cosine sim)
        # We generally want visually compatible images
        emb1 = img1["embedding"]
        emb2 = img2["embedding"]
        if hasattr(emb1, "numpy"): emb1 = emb1.numpy()
        if hasattr(emb2, "numpy"): emb2 = emb2.numpy()
        
        # Flatten if needed
        if len(emb1.shape) > 1: emb1 = emb1.flatten()
        if len(emb2.shape) > 1: emb2 = emb2.flatten()
        
        # Cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 > 0 and norm2 > 0:
            sim = np.dot(emb1, emb2) / (norm1 * norm2)
        else:
            sim = 0.0
            
        # 3. Hybrid Score
        if self.pairing_strategy == "tag":
            return float(tag_score)
        elif self.pairing_strategy == "embedding":
            return float(sim)
        else: # hybrid
            # Normalize tag score roughly (e.g., max 20 tags)
            norm_tag_score = min(tag_score / 20.0, 1.0)
            return (0.4 * norm_tag_score) + (0.6 * sim)
