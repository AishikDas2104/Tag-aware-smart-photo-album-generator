import sys
print(f"Python version: {sys.version}")

try:
    import numpy
    print(f"numpy: {numpy.__version__}")
except ImportError as e:
    print(f"numpy error: {e}")

try:
    import torch
    print(f"torch: {torch.__version__}")
except ImportError as e:
    print(f"torch error: {e}")

try:
    import PIL
    print(f"PIL: {PIL.__version__}")
except ImportError as e:
    print(f"PIL error: {e}")

try:
    import open_clip
    print(f"open_clip: {open_clip.__version__}")
except ImportError as e:
    print(f"open_clip error: {e}")

try:
    import faiss
    print(f"faiss: {faiss.__version__ if hasattr(faiss, '__version__') else 'installed'}")
except ImportError as e:
    print(f"faiss error: {e}")

try:
    import moviepy
    print(f"moviepy: {moviepy.__version__}")
except ImportError as e:
    print(f"moviepy error: {e}")
