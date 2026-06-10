"""
Sephora dataset parser — converts Kaggle CSV into our product schema
and filters to skincare-relevant categories.

Usage:
    cd backend
    python -m data.parse_sephora         # parse + export to data/sephora_products.json
    python -m data.parse_sephora --stats  # show category distribution
"""

import ast
import csv
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

SEPHORA_CSV = DATA_DIR / "sephora" / "product_info.csv"
OUTPUT_PATH = DATA_DIR / "sephora_products.json"

# Categories we want for skincare routines
SKINCARE_CATEGORIES = {
    "Skincare", "Face Wash & Cleansers", "Moisturizers", "Lip Treatments",
    "Sunscreen", "Face Serums", "Face Oils", "Eye Cream & Eye Treatments",
    "Facial Peels", "Face Masks", "Toners", "Mists & Essences",
    "Moisturizer", "Cleanser", "Treatment", "Face Primer",
}

# Map Sephora categories → our routine steps
SEPHORA_CATEGORY_MAP = {
    "face wash": "Cleanser",
    "cleanser": "Cleanser",
    "cleansing": "Cleanser",
    "makeup remover": "Cleanser",
    "toner": "Toner",
    "mist": "Toner",
    "essence": "Toner",
    "serum": "Serum",
    "ampoule": "Serum",
    "treatment": "Treatment",
    "peel": "Treatment",
    "exfoliat": "Treatment",
    "retinol": "Treatment",
    "acne": "Treatment",
    "moisturizer": "Moisturizer",
    "cream": "Moisturizer",
    "lotion": "Moisturizer",
    "oil": "Face Oil",
    "sunscreen": "Sunscreen",
    "spf": "Sunscreen",
    "sun protection": "Sunscreen",
    "eye cream": "Eye Cream",
    "eye treatment": "Eye Cream",
    "lip": "Lip Care",
    "mask": "Mask",
}


def _classify_category(row: dict) -> str | None:
    """Map Sephora's category hierarchy to our simplified category."""
    combined = " ".join([
        row.get("primary_category", ""),
        row.get("secondary_category", ""),
        row.get("tertiary_category", ""),
    ]).lower()

    for keyword, category in SEPHORA_CATEGORY_MAP.items():
        if keyword in combined:
            return category
    return None


def _parse_ingredients(raw: str) -> list[str]:
    """Extract clean ingredient names from Sephora's ingredient string."""
    if not raw or raw.strip() in ("", "[]", "N/A"):
        return []

    # Try parsing as a Python list literal first
    try:
        items = ast.literal_eval(raw)
        if isinstance(items, list):
            # Clean each ingredient
            cleaned = []
            for item in items[:20]:  # cap at 20 key ingredients
                # Remove parenthetical INCI names, numbers, etc.
                name = re.sub(r'\([^)]*\)', '', str(item)).strip()
                name = re.sub(r'\s+', ' ', name).strip(' .,;:-')
                if name and len(name) > 2:
                    cleaned.append(name)
            return cleaned
    except (ValueError, SyntaxError):
        pass

    # Fallback: split by comma
    parts = raw.strip("[]'\"").split(",")
    return [p.strip() for p in parts[:20] if p.strip() and len(p.strip()) > 2]


def _parse_highlights(raw: str) -> tuple[list[str], list[str]]:
    """Extract skin types and concerns from Sephora highlights."""
    skin_types = []
    concerns = []

    try:
        items = ast.literal_eval(raw) if raw else []
    except (ValueError, SyntaxError):
        items = []

    for item in items:
        item_lower = str(item).lower()
        if "good for:" in item_lower:
            concern = item_lower.replace("good for:", "").strip()
            concerns.append(concern)
        elif "oily" in item_lower:
            skin_types.append("oily")
        elif "dry" in item_lower:
            skin_types.append("dry")
        elif "combination" in item_lower:
            skin_types.append("combination")
        elif "sensitive" in item_lower:
            skin_types.append("sensitive")
        elif "normal" in item_lower:
            skin_types.append("normal")
        elif "hydrating" in item_lower:
            concerns.append("hydration")
        elif "anti-aging" in item_lower or "aging" in item_lower:
            concerns.append("anti-aging")
        elif "acne" in item_lower:
            concerns.append("acne")
        elif "brightening" in item_lower:
            concerns.append("brightening")
        elif "pore" in item_lower:
            concerns.append("pores")

    return skin_types or ["all"], concerns or ["general skincare"]


def parse_sephora(show_stats: bool = False) -> list[dict]:
    """Parse the Sephora CSV into our product schema."""
    if not SEPHORA_CSV.exists():
        logger.error("Sephora CSV not found at %s", SEPHORA_CSV)
        logger.error("Download: kaggle datasets download -d nadyinky/sephora-products-and-skincare-reviews")
        return []

    products = []
    skipped = 0
    category_counts: dict[str, int] = {}

    with open(SEPHORA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = _classify_category(row)
            if not category:
                skipped += 1
                continue

            category_counts[category] = category_counts.get(category, 0) + 1

            # Parse price
            try:
                price = float(row.get("price_usd", 0))
            except (ValueError, TypeError):
                price = 0.0

            if price <= 0:
                skipped += 1
                continue

            ingredients = _parse_ingredients(row.get("ingredients", ""))
            skin_types, concerns = _parse_highlights(row.get("highlights", ""))

            product = {
                "id": row["product_id"].strip(),
                "name": row["product_name"].strip(),
                "brand": row["brand_name"].strip(),
                "category": category,
                "price": price,
                "rating": round(float(row.get("rating", 0) or 0), 2),
                "key_ingredients": ingredients,
                "skin_types": skin_types,
                "concerns": concerns,
                "description": f"{row['product_name']} by {row['brand_name']}. {category}.",
            }
            products.append(product)

    logger.info("Parsed %d skincare products (%d skipped)", len(products), skipped)

    if show_stats:
        logger.info("Category distribution:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            logger.info("  %-15s %d", cat, count)

    return products


def main():
    show_stats = "--stats" in sys.argv
    products = parse_sephora(show_stats=show_stats)

    if products:
        with open(OUTPUT_PATH, "w") as f:
            json.dump(products, f, indent=2)
        logger.info("Saved %d products to %s", len(products), OUTPUT_PATH)


if __name__ == "__main__":
    main()
