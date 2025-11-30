import numpy as np
import faiss
from typing import List, Tuple, Dict
import logging
from ..slide_builder.slide_constructor import Slide

logger = logging.getLogger(__name__)

class EmbeddingIndex:
    """
    FAISS-based index for efficient similarity search on slide embeddings.
    """
    
    def __init__(self, dimension: int = None):
        self.dimension = dimension  # Will be set from data if None
        self.index = None
        self.slide_id_map: Dict[int, int] = {} # FAISS internal ID -> Slide ID
        self.reverse_map: Dict[int, int] = {}  # Slide ID -> FAISS internal ID
        
    def build(self, slides: List[Slide]):
        """
        Builds the FAISS index from slide embeddings.
        """
        if not slides:
            return
            
        # Prepare data
        embeddings = []
        ids = []
        
        for i, slide in enumerate(slides):
            emb = slide.embedding
            # Ensure correct shape and type
            if len(emb.shape) == 1:
                emb = emb.reshape(1, -1)
            
            embeddings.append(emb)
            self.slide_id_map[i] = slide.id
            self.reverse_map[slide.id] = i
            ids.append(i)
            
        # Concatenate all embeddings
        data = np.vstack(embeddings).astype('float32')
        
        # Auto-detect dimension if not set
        if self.dimension is None:
            self.dimension = data.shape[1]
            logger.info(f"Auto-detected embedding dimension: {self.dimension}")
        
        # Verify dimension matches
        if data.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {data.shape[1]}")
        
        # Normalize for cosine similarity (Inner Product)
        faiss.normalize_L2(data)
        
        # Choose index type based on dataset size
        n_samples = data.shape[0]
        
        if n_samples < 10000:
            # Exact search for smaller datasets
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(data)
            logger.info(f"Built exact FAISS index for {n_samples} slides with dimension {self.dimension}.")
        else:
            # Approximate search for larger datasets
            nlist = min(int(4 * np.sqrt(n_samples)), n_samples // 39)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(data)
            self.index.add(data)
            logger.info(f"Built approximate FAISS index (IVF) for {n_samples} slides with dimension {self.dimension}.")

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Finds k nearest neighbors for the query embedding.
        Returns list of (slide_id, score).
        """
        if self.index is None:
            return []
            
        # Prepare query
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        q_data = query_embedding.astype('float32')
        faiss.normalize_L2(q_data)
        
        # Search
        scores, indices = self.index.search(q_data, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1: # FAISS returns -1 if not enough neighbors found
                slide_id = self.slide_id_map.get(idx)
                if slide_id is not None:
                    results.append((slide_id, float(score)))
                    
        return results
