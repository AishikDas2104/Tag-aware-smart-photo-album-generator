import argparse
import logging
from pathlib import Path
from src.web_ui.app import app

def main():
    parser = argparse.ArgumentParser(description="Tag-Aware Photo Story Generator")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of Web UI")
    parser.add_argument("--input", type=str, help="Input directory containing images")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    
    args = parser.parse_args()
    
    if args.cli:
        if not args.input:
            print("Error: --input is required for CLI mode")
            return
            
        print(f"Starting CLI processing for {args.input}...")
        # TODO: Call pipeline directly
        print("CLI mode not fully implemented in this entry point yet. Please use the Web UI or implement direct pipeline call.")
    else:
        print("Launching Web UI...")
        app.launch()

if __name__ == "__main__":
    main()
