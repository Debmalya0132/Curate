"""
Central configuration for the Pinterest Visual Search backend.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
IMAGES_DIR = DATA_DIR / "images"
SEED_PRODUCTS_PATH = DATA_DIR / "seed_products.json"

# ── Model ──────────────────────────────────────────────────────────
SIGLIP_MODEL = "google/siglip-base-patch16-224"
EMBEDDING_DIM = 768

# ── Search ─────────────────────────────────────────────────────────
DEFAULT_TOP_K = 12
MAX_TOP_K = 50

# ── API ────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://localhost"
).split(",")

# ── Auto-create directories ───────────────────────────────────────
INDEX_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
