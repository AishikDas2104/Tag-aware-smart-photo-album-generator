from typing import List, Dict, Set, Any
from collections import defaultdict
import logging
from ..slide_builder.slide_constructor import Slide

logger = logging.getLogger(__name__)

class TagIndex:
    """
    Inverted index mapping tags to slides.
    Allows fast retrieval of slides containing specific tags.
    """
    
    def __init__(self):
        self.index: Dict[str, List[int]] = defaultdict(list)
        self.slides: Dict[int, Slide] = {}
        
    def build(self, slides: List[Slide]):
        """
        Builds the inverted index from a list of slides.
        """
        self.index.clear()
        self.slides.clear()
        
        for slide in slides:
            self.slides[slide.id] = slide
            for tag in slide.tags:
                self.index[tag].append(slide.id)
                
        logger.info(f"Built tag index with {len(self.index)} unique tags covering {len(slides)} slides.")

    def get_slides_by_tag(self, tag: str) -> List[Slide]:
        """
        Returns all slides containing the given tag.
        """
        slide_ids = self.index.get(tag, [])
        return [self.slides[sid] for sid in slide_ids]

    def get_slides_by_tags(self, tags: List[str], mode: str = "union") -> List[Slide]:
        """
        Returns slides matching a list of tags.
        mode: "union" (OR) or "intersection" (AND)
        """
        if not tags:
            return []
            
        result_ids = set()
        first = True
        
        for tag in tags:
            ids = set(self.index.get(tag, []))
            
            if mode == "intersection":
                if first:
                    result_ids = ids
                    first = False
                else:
                    result_ids &= ids
            else: # union
                result_ids |= ids
                
        return [self.slides[sid] for sid in result_ids]
