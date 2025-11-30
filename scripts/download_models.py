"""
Model Download Helper
Pre-downloads CLIP models to avoid timeouts in the Web UI.
"""
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_name, pretrained):
    """Downloads and caches a CLIP model."""
    try:
        import open_clip
        logger.info(f"Downloading {model_name} ({pretrained})...")
        logger.info("This may take several minutes on first run...")
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained
        )
        
        logger.info(f"✓ {model_name} downloaded and cached successfully!")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CLIP Model Downloader")
    print("="*60 + "\n")
    
    models = {
        "ViT-B-32": "laion2b_s34b_b79k",  # ~600MB
        "ViT-L-14": "laion2b_s32b_b82k",  # ~1.7GB
    }
    
    print("Available models:")
    for i, (name, pretrained) in enumerate(models.items(), 1):
        size = "600MB" if name == "ViT-B-32" else "1.7GB"
        print(f"  {i}. {name} ({size})")
    
    choice = input("\nSelect model to download (1 or 2, or 'all'): ").strip().lower()
    
    to_download = []
    if choice == "all":
        to_download = list(models.items())
    elif choice in ["1", "2"]:
        to_download = [list(models.items())[int(choice)-1]]
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    print()
    for name, pretrained in to_download:
        success = download_model(name, pretrained)
        if not success:
            sys.exit(1)
    
    print("\n" + "="*60)
    print("All models downloaded successfully!")
    print("You can now use the Web UI without download delays.")
    print("="*60 + "\n")
