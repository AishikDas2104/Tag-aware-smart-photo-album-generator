import json
import csv
from typing import List, Dict, Any
from pathlib import Path
import logging
from datetime import datetime
import numpy as np

from ..config import Config
from ..slide_builder.slide_constructor import Slide

logger = logging.getLogger(__name__)

class OutputGenerator:
    """
    Generates final outputs: JSON, CSV, and Video.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_json(self, slides: List[Slide], filename: str = "story.json"):
        """
        Saves the story sequence to a JSON file.
        """
        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_slides": len(slides),
                "total_images": sum(len(s.image_ids) for s in slides)
            },
            "sequence": []
        }
        
        for i, slide in enumerate(slides):
            slide_data = {
                "sequence_id": i,
                "slide_id": slide.id,
                "images": slide.image_ids,
                "tags": list(slide.tags),
                "orientation": slide.orientation,
                "metadata": {k: str(v) for k, v in slide.metadata.items()} # Ensure serializable
            }
            output_data["sequence"].append(slide_data)
            
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Saved JSON story to {output_path}")
        return str(output_path)

    def save_csv(self, slides: List[Slide], filename: str = "story.csv"):
        """
        Saves the story sequence to a CSV file.
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Sequence ID", "Slide ID", "Images", "Tags", "Orientation", "Timestamp"])
            
            for i, slide in enumerate(slides):
                writer.writerow([
                    i,
                    slide.id,
                    " | ".join(slide.image_ids),
                    ", ".join(list(slide.tags)),
                    slide.orientation,
                    slide.metadata.get("timestamp", "")
                ])
                
        logger.info(f"Saved CSV story to {output_path}")
        return str(output_path)

    def generate_video(self, slides: List[Slide], filename: str = "story.mp4"):
        """
        Generates a video slideshow using MoviePy.
        """
        if not Config.VIDEO_ENABLED:
            logger.info("Video generation disabled in config.")
            return None
            
        try:
            from moviepy.editor import ImageClip, concatenate_videoclips, CompositeVideoClip, TextClip
            from PIL import Image
        except ImportError:
            logger.warning("MoviePy not installed. Skipping video generation.")
            return None
            
        logger.info("Generating video slideshow...")
        clips = []
        
        # Video settings
        w, h = 1920, 1080  # Full HD
        duration = Config.SLIDE_DURATION
        transition_duration = Config.TRANSITION_DURATION
        
        for slide in slides:
            # Create image clip
            if len(slide.image_ids) == 1:
                # Single image
                img_path = slide.image_ids[0]
                clip = self._create_image_clip(img_path, w, h, duration)
            else:
                # Paired images (side-by-side)
                clip = self._create_paired_clip(slide.image_ids, w, h, duration)
                
            # Add fadein/fadeout for smooth transitions
            clip = clip.crossfadein(transition_duration)
            clips.append(clip)
            
        # Concatenate
        # method='compose' handles overlapping for crossfades
        final_video = concatenate_videoclips(clips, method="compose", padding=-transition_duration)
        
        output_path = str(self.output_dir / filename)
        final_video.write_videofile(output_path, fps=24, codec='libx264', audio=False)
        
        logger.info(f"Saved video story to {output_path}")
        return output_path

    def _create_image_clip(self, img_path, w, h, duration):
        from moviepy.editor import ImageClip
        
        clip = ImageClip(img_path).set_duration(duration)
        
        # Resize to fit screen (contain)
        img_w, img_h = clip.size
        ratio = min(w / img_w, h / img_h)
        new_w, new_h = int(img_w * ratio), int(img_h * ratio)
        
        clip = clip.resize((new_w, new_h))
        
        # Center on black background
        clip = clip.set_position(("center", "center"))
        
        # Create composite with black background
        # For simplicity in this snippet, we return the resized clip
        # Ideally we'd composite it over a black background of size w,h
        return clip

    def _create_paired_clip(self, img_paths, w, h, duration):
        from moviepy.editor import ImageClip, clips_array
        
        # Load both images
        clip1 = ImageClip(img_paths[0]).set_duration(duration)
        clip2 = ImageClip(img_paths[1]).set_duration(duration)
        
        # Resize to fit half screen width
        target_w = w // 2
        
        clip1 = clip1.resize(width=target_w)
        clip2 = clip2.resize(width=target_w)
        
        # Combine side-by-side
        final_clip = clips_array([[clip1, clip2]])
        
        # Resize if height exceeds screen
        if final_clip.h > h:
            final_clip = final_clip.resize(height=h)
            
        return final_clip.set_position(("center", "center"))
