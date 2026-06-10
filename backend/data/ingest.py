"""
Data ingestion pipeline — builds the FAISS search index from the product catalog.

Modes:
  Text-based (default):  Encodes rich product descriptions with SigLIP.
  Image-based:           Encodes product images from data/images/<id>.jpg.

Usage:
    cd backend
    python -m data.ingest                # text mode (no images required)
    python -m data.ingest --use-images   # image mode
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, INDEX_DIR, IMAGES_DIR, SEED_PRODUCTS_PATH, EMBEDDING_DIM
from retrieval.clip_encoder import SigLIPEngine
from retrieval.faiss_index import VectorIndex

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def _description(p: dict) -> str:
    """Build a rich search-friendly text from product metadata."""
    return ". ".join([
        f"{p['name']} by {p['brand']}",
        f"Category: {p['category']}",
        f"For {', '.join(p['skin_types'])} skin",
        f"Targets: {', '.join(p['concerns'])}",
        f"Key ingredients: {', '.join(p['key_ingredients'][:5])}",
        p.get("description", ""),
    ])


def ingest_text(products, encoder, index):
    """Index products via text descriptions (no images needed)."""
    descs = [_description(p) for p in products]
    ids = [p["id"] for p in products]

    bs = 16
    chunks = []
    for i in range(0, len(descs), bs):
        batch = descs[i : i + bs]
        chunks.append(encoder.encode_texts(batch))
        logger.info("  batch %d/%d", i // bs + 1, (len(descs) + bs - 1) // bs)

    index.add(np.vstack(chunks), ids)


def ingest_images(products, encoder, index):
    """Index products via image files in data/images/."""
    paths, ids = [], []
    for p in products:
        for ext in ("jpg", "png", "webp"):
            fp = IMAGES_DIR / f"{p['id']}.{ext}"
            if fp.exists():
                paths.append(fp)
                ids.append(p["id"])
                break
        else:
            logger.warning("  no image for %s — skipping", p["id"])

    if not paths:
        logger.error("No images found in %s", IMAGES_DIR)
        return

    bs = 8
    chunks = []
    for i in range(0, len(paths), bs):
        chunks.append(encoder.encode_images(paths[i : i + bs]))
        logger.info("  batch %d/%d", i // bs + 1, (len(paths) + bs - 1) // bs)

    index.add(np.vstack(chunks), ids)


def main():
    use_images = "--use-images" in sys.argv
    use_sephora = "--sephora" in sys.argv

    if use_sephora:
        sephora_path = DATA_DIR / "sephora_products.json"
        if not sephora_path.exists():
            logger.error("Run `python -m data.parse_sephora` first")
            sys.exit(1)
        with open(sephora_path) as f:
            products = json.load(f)
    else:
        with open(SEED_PRODUCTS_PATH) as f:
            products = json.load(f)
    logger.info("Loaded %d products", len(products))

    encoder = SigLIPEngine()
    idx = VectorIndex(dim=EMBEDDING_DIM)

    if use_images:
        ingest_images(products, encoder, idx)
    else:
        ingest_text(products, encoder, idx)

    idx.save(INDEX_DIR)

    # Persist product metadata for API lookups
    meta = {p["id"]: p for p in products}
    with open(INDEX_DIR / "products.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Done — %d vectors in index at %s", idx.size, INDEX_DIR)


if __name__ == "__main__":
    main()

