import gradio as gr
import os
import shutil
from pathlib import Path
import logging
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import Config
from src.input_processor.input_processor import InputProcessor
from src.feature_extractor.clip_extractor import CLIPExtractor
from src.slide_builder.slide_constructor import SlideConstructor
from src.indexing.tag_index import TagIndex
from src.indexing.embedding_index import EmbeddingIndex
from src.sequencing.sequence_optimizer import SequenceOptimizer
from src.output_generator.generator import OutputGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import traceback

def check_model_available(model_name, pretrained):
    """Check if model is already downloaded."""
    try:
        import open_clip
        cache_dir = Path.home() / ".cache" / "open_clip"
        # Try to load model metadata
        open_clip.get_model_config(model_name)
        return True
    except:
        return False

def generate_story(input_path, output_path, model_selection, pairing_strategy, w_tags, w_emb, w_time):
    """
    Main pipeline function triggered by UI.
    Returns: (status_text, json_file, csv_file, video_file)
    """
    try:
        logger.info(f"Starting generation with input: {input_path}")
        
        # Validate input path
        if not os.path.exists(input_path):
            error_msg = f"Error: Input directory does not exist: {input_path}"
            logger.error(error_msg)
            return error_msg, None, None, None
        
        # Update config based on UI inputs
        Config.CLIP_MODEL_NAME = model_selection
        Config.CLIP_PRETRAINED = Config.MODEL_MAPPING[model_selection]
        
        Config.PAIRING_STRATEGY = pairing_strategy
        Config.W_TAGS = w_tags
        Config.W_EMBEDDING = w_emb
        Config.W_TEMPORAL = w_time
        
        # Create output dir
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        status = f"Step 1/6: Processing Input Images (Using {model_selection})...\n"
        logger.info(status)
        
        processor = InputProcessor(input_path)
        
        # 1. Metadata
        all_metadata = []
        for batch in processor.process_images(batch_size=32):
            all_metadata.extend(batch)
        
        status += f"Found {len(all_metadata)} images.\n"
        logger.info(f"Found {len(all_metadata)} images")
            
        status += "Step 2/6: Extracting Features (CLIP)...\n"
        status += "(This may take a few minutes on first run while downloading model)\n"
        logger.info("Initializing CLIP extractor")
        
        extractor = CLIPExtractor()
        
        processed_data = []
        batch_size = 32
        
        for i in range(0, len(all_metadata), batch_size):
            batch_meta = all_metadata[i:i+batch_size]
            batch_paths = [m["filepath"] for m in batch_meta]
            
            loaded = processor.load_and_preprocess_batch(batch_paths)
            if not loaded:
                continue
                
            imgs = [x[1] for x in loaded]
            
            # Extract features
            embeddings, tags_batch = extractor.process_batch(imgs)
            
            for j, (fp, img) in enumerate(loaded):
                meta = batch_meta[j]
                # Restructure to match SlideConstructor expectations
                processed_data.append({
                    "filepath": meta["filepath"],
                    "tags": tags_batch[j],
                    "embedding": embeddings[j],
                    "orientation": meta.get("orientation", "landscape"),
                    "metadata": {
                        "timestamp": meta.get("timestamp"),
                        "gps_coords": meta.get("gps_coords"),
                        "width": meta.get("width"),
                        "height": meta.get("height"),
                        "hash": meta.get("hash"),
                        "camera_model": meta.get("camera_model")
                    }
                })
            
            progress = f"Processed {min(i+batch_size, len(all_metadata))}/{len(all_metadata)} images\n"
            status += progress
            logger.info(progress.strip())
            
        status += "Step 3/6: Building Slides...\n"
        logger.info("Building slides")
        
        constructor = SlideConstructor(pairing_strategy=pairing_strategy)
        slides = constructor.build_slides(processed_data)
        
        status += f"Created {len(slides)} slides.\n"
        logger.info(f"Created {len(slides)} slides")
        
        status += "Step 4/6: Indexing...\n"
        logger.info("Building indexes")
        
        tag_index = TagIndex()
        tag_index.build(slides)
        
        emb_index = EmbeddingIndex()
        emb_index.build(slides)
        
        status += "Step 5/6: Optimizing Sequence...\n"
        logger.info("Optimizing sequence")
        
        optimizer = SequenceOptimizer(tag_index, emb_index)
        ordered_slides = optimizer.optimize(slides)
        
        status += "Step 6/6: Generating Outputs...\n"
        logger.info("Generating outputs")
        
        gen = OutputGenerator(output_path)
        json_path = gen.save_json(ordered_slides)
        csv_path = gen.save_csv(ordered_slides)
        
        video_path = None
        if Config.VIDEO_ENABLED:
            status += "Generating video (this may take a while)...\n"
            logger.info("Generating video")
            video_path = gen.generate_video(ordered_slides)
            
        status += f"\n✅ Done! Generated story with {len(ordered_slides)} slides.\n"
        status += f"JSON: {json_path}\n"
        status += f"CSV: {csv_path}\n"
        if video_path:
            status += f"Video: {video_path}\n"
        
        logger.info("Generation complete!")
        return status, json_path, csv_path, video_path
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nSee app.log for full details."
        logger.error(f"Error during generation: {e}")
        logger.error(traceback.format_exc())
        return error_msg, None, None, None

# UI Definition
with gr.Blocks(title="Tag-Aware Photo Story Generator") as app:
    gr.Markdown("# 📸 Tag-Aware Photo Story Generator")
    gr.Markdown("Transform your photo collection into a cinematic story.")
    
    with gr.Row():
        with gr.Column():
            input_path = gr.Textbox(label="Input Image Directory", placeholder="C:/path/to/images")
            output_path = gr.Textbox(label="Output Directory", value="output")
            
            with gr.Accordion("Advanced Settings", open=True):
                model_selection = gr.Dropdown(
                    choices=list(Config.MODEL_MAPPING.keys()),
                    value="ViT-L-14",
                    label="AI Model (Quality vs Speed)"
                )
                pairing_strategy = gr.Dropdown(
                    choices=["hybrid", "tag", "embedding"], 
                    value="hybrid", 
                    label="Pairing Strategy"
                )
                w_tags = gr.Slider(0, 5, value=1.0, label="Tag Weight")
                w_emb = gr.Slider(0, 5, value=2.0, label="Visual Similarity Weight")
                w_time = gr.Slider(0, 5, value=0.5, label="Time Continuity Weight")
            
            generate_btn = gr.Button("Generate Story", variant="primary")
            
        with gr.Column():
            status_output = gr.Textbox(label="Status", interactive=False)
            json_output = gr.File(label="JSON Sequence")
            csv_output = gr.File(label="CSV Sequence")
            video_output = gr.Video(label="Generated Video")
            
    generate_btn.click(
        generate_story,
        inputs=[input_path, output_path, model_selection, pairing_strategy, w_tags, w_emb, w_time],
        outputs=[status_output, json_output, csv_output, video_output]
    )

if __name__ == "__main__":
    app.launch()
