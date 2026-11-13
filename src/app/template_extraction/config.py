"""Settings for the template-guided extraction example."""

import os
from pathlib import Path

CHAT_MODEL = os.getenv("EXTRACT_CHAT_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
NUM_CTX = 8192  # chunk prompts plus their JSON output must fit

PACKAGE_DIR = Path(__file__).parent
REGISTRY_DIR = PACKAGE_DIR / "registry"
SAMPLES_DIR = PACKAGE_DIR / "samples"
OUT_DIR = PACKAGE_DIR / "out"  # template candidates for unknown vendors (git-ignored)

# Line-item chunking. Small chunks keep a 7B model accurate; the overlap lets
# stitching catch rows that a chunk boundary would otherwise cut.
CHUNK_ROWS = 15
CHUNK_OVERLAP = 2

# Gap between words (in PDF points) treated as a column boundary in the text layer.
COLUMN_GAP = 10.0

MONEY_TOLERANCE = 0.01
