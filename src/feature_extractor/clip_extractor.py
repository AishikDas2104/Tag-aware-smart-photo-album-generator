import torch
import open_clip
from PIL import Image
from typing import List, Dict, Any, Tuple
import logging
from ..config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIPExtractor:
    """
    Extracts semantic embeddings and tags using OpenAI's CLIP model.
    """
    
    # Comprehensive tag vocabulary for zero-shot classification
    TAG_VOCABULARY = [
        # Nature & Landscapes
        "beach", "ocean", "sea", "waves", "sunset", "sunrise", "mountains", "forest", 
        "trees", "lake", "river", "waterfall", "desert", "sky", "clouds", "stars", 
        "night", "snow", "winter", "autumn", "summer", "spring", "flowers", "garden",
        "park", "field", "grass", "rocks", "canyon", "valley", "island", "coast",
        
        # Urban & Architecture
        "city", "skyline", "building", "skyscraper", "street", "road", "bridge", 
        "house", "apartment", "interior", "room", "window", "door", "stairs", 
        "wall", "ceiling", "floor", "furniture", "kitchen", "bedroom", "bathroom", 
        "living room", "office", "library", "museum", "church", "temple", "castle",
        
        # People & Activities
        "person", "people", "man", "woman", "child", "baby", "crowd", "group", 
        "portrait", "selfie", "smile", "happy", "party", "wedding", "concert", 
        "festival", "sports", "running", "swimming", "dancing", "eating", "drinking",
        "sleeping", "working", "playing", "travel", "vacation", "hiking",
        
        # Animals
        "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "wildlife", "zoo", 
        "pet", "insect", "butterfly",
        
        # Objects & Things
        "car", "bus", "train", "airplane", "boat", "bicycle", "motorcycle", 
        "food", "drink", "coffee", "cake", "fruit", "vegetable", "book", "phone", 
        "computer", "camera", "art", "painting", "sculpture", "toy", "clothes", 
        "shoes", "bag", "glasses", "watch", "jewelry",
        
        # Abstract & Artistic
        "colorful", "black and white", "monochrome", "bright", "dark", "shadow", 
        "light", "reflection", "texture", "pattern", "minimalist", "vintage", 
        "retro", "abstract", "blur", "bokeh"
    ]

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing CLIP model on {self.device}...")
        
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                Config.CLIP_MODEL_NAME, 
                pretrained=Config.CLIP_PRETRAINED,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(Config.CLIP_MODEL_NAME)
            
            # Pre-compute text embeddings for tags
            self.text_features = self._precompute_tag_embeddings()
            logger.info("CLIP model initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize CLIP: {e}")
            raise

    def _precompute_tag_embeddings(self) -> torch.Tensor:
        """
        Computes embeddings for all tags in the vocabulary once during initialization.
        """
        logger.info("Pre-computing tag embeddings...")
        text_tokens = self.tokenizer(self.TAG_VOCABULARY).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features

    def get_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Generates a normalized embedding for a single image.
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu()

    def get_tags(self, image_features: torch.Tensor, threshold: float = Config.TAG_THRESHOLD) -> List[str]:
        """
        Returns a list of tags that match the image above the given threshold.
        Expects image_features to be on the same device as text_features (or handled via move).
        """
        # Ensure image_features is on the correct device
        if image_features.device != self.device:
            image_features = image_features.to(self.device)
            
        # Compute similarity
        # image_features: [1, 768], text_features: [N_tags, 768]
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(10) # Get top 10 candidates first
        
        # Filter by raw cosine similarity or softmax probability? 
        # CLIP logit_scale is high, so softmax is very peaked. 
        # Let's use raw cosine similarity for thresholding if possible, 
        # but open_clip encode_image returns normalized features.
        # Let's calculate raw cosine similarity manually for thresholding.
        
        raw_similarity = image_features @ self.text_features.T
        
        relevant_tags = []
        for i in range(len(self.TAG_VOCABULARY)):
            score = raw_similarity[0][i].item()
            if score > threshold:
                relevant_tags.append(self.TAG_VOCABULARY[i])
                
        # Sort by score descending
        relevant_tags.sort(key=lambda t: raw_similarity[0][self.TAG_VOCABULARY.index(t)].item(), reverse=True)
        
        return relevant_tags

    def process_batch(self, images: List[Image.Image]) -> Tuple[torch.Tensor, List[List[str]]]:
        """
        Process a batch of images to get embeddings and tags.
        """
        processed_images = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(processed_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        # Get tags for each image
        batch_tags = []
        raw_similarities = image_features @ self.text_features.T
        
        for i in range(len(images)):
            tags = []
            for j in range(len(self.TAG_VOCABULARY)):
                if raw_similarities[i][j].item() > Config.TAG_THRESHOLD:
                    tags.append(self.TAG_VOCABULARY[j])
            # Sort
            tags.sort(key=lambda t: raw_similarities[i][self.TAG_VOCABULARY.index(t)].item(), reverse=True)
            batch_tags.append(tags)
            
        return image_features.cpu(), batch_tags
