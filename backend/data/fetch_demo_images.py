"""
Download a curated set of real skincare product images from public brand CDNs
for cross-modal search demos. These are NOT Sephora images — they are from
the brands' own public product pages.

Usage:
    cd backend
    python -m data.fetch_demo_images
"""

import logging
import sys
import ssl
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMAGES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Curated list: (filename, public image URL)
# Sources: brand product pages, Unsplash, Pexels, open product databases
DEMO_IMAGES = [
    # The Ordinary — public product images from DECIEM CDN
    ("ordinary_niacinamide.jpg",
     "https://www.theordinary.com/_next/image?url=https%3A%2F%2Fimages-us-ias.thenewblack.ai%2FWWW_THEORDINARY%2Fthumbnails%2Foptimized%2Ffull-size%2FRDN-Niacinamide-10pct-Zinc-1pct-30ml.png&w=640&q=70"),
    ("ordinary_ha.jpg",
     "https://www.theordinary.com/_next/image?url=https%3A%2F%2Fimages-us-ias.thenewblack.ai%2FWWW_THEORDINARY%2Fthumbnails%2Foptimized%2Ffull-size%2FRDN-Hyaluronic-Acid-2pct-B5-30ml.png&w=640&q=70"),
    ("ordinary_retinol.jpg",
     "https://www.theordinary.com/_next/image?url=https%3A%2F%2Fimages-us-ias.thenewblack.ai%2FWWW_THEORDINARY%2Fthumbnails%2Foptimized%2Ffull-size%2FRDN-Retinol-0.5pct-in-Squalane-30ml.png&w=640&q=70"),
    ("ordinary_aha_bha.jpg",
     "https://www.theordinary.com/_next/image?url=https%3A%2F%2Fimages-us-ias.thenewblack.ai%2FWWW_THEORDINARY%2Fthumbnails%2Foptimized%2Ffull-size%2FRDN-AHA-30pct-BHA-2pct-Peeling-Solution-30ml.png&w=640&q=70"),
    ("ordinary_squalane.jpg",
     "https://www.theordinary.com/_next/image?url=https%3A%2F%2Fimages-us-ias.thenewblack.ai%2FWWW_THEORDINARY%2Fthumbnails%2Foptimized%2Ffull-size%2FRDN-100pct-Plant-Derived-Squalane-30ml.png&w=640&q=70"),
    ("ordinary_vitc.jpg",
     "https://www.theordinary.com/_next/image?url=https%3A%2F%2Fimages-us-ias.thenewblack.ai%2FWWW_THEORDINARY%2Fthumbnails%2Foptimized%2Ffull-size%2FRDN-Vitamin-C-Suspension-23pct-HA-Spheres-2pct-30ml.png&w=640&q=70"),
    # General skincare product photos from Pexels (CC0 licensed)
    ("demo_serum_bottle.jpg",
     "https://images.pexels.com/photos/3785147/pexels-photo-3785147.jpeg?auto=compress&cs=tinysrgb&w=600"),
    ("demo_cream_jar.jpg",
     "https://images.pexels.com/photos/3018845/pexels-photo-3018845.jpeg?auto=compress&cs=tinysrgb&w=600"),
    ("demo_skincare_bottles.jpg",
     "https://images.pexels.com/photos/3735149/pexels-photo-3735149.jpeg?auto=compress&cs=tinysrgb&w=600"),
    ("demo_face_oil.jpg",
     "https://images.pexels.com/photos/4041392/pexels-photo-4041392.jpeg?auto=compress&cs=tinysrgb&w=600"),
    ("demo_sunscreen.jpg",
     "https://images.pexels.com/photos/5217959/pexels-photo-5217959.jpeg?auto=compress&cs=tinysrgb&w=600"),
    ("demo_cleanser.jpg",
     "https://images.pexels.com/photos/6621462/pexels-photo-6621462.jpeg?auto=compress&cs=tinysrgb&w=600"),
    ("demo_moisturizer.jpg",
     "https://images.pexels.com/photos/5797999/pexels-photo-5797999.jpeg?auto=compress&cs=tinysrgb&w=600"),
    ("demo_routine_flat.jpg",
     "https://images.pexels.com/photos/3685530/pexels-photo-3685530.jpeg?auto=compress&cs=tinysrgb&w=600"),
    ("demo_dropper_serum.jpg",
     "https://images.pexels.com/photos/4465124/pexels-photo-4465124.jpeg?auto=compress&cs=tinysrgb&w=600"),
    ("demo_toner.jpg",
     "https://images.pexels.com/photos/7797086/pexels-photo-7797086.jpeg?auto=compress&cs=tinysrgb&w=600"),
    ("demo_eye_cream.jpg",
     "https://images.pexels.com/photos/6621466/pexels-photo-6621466.jpeg?auto=compress&cs=tinysrgb&w=600"),
    ("demo_mask.jpg",
     "https://images.pexels.com/photos/3737597/pexels-photo-3737597.jpeg?auto=compress&cs=tinysrgb&w=600"),
    ("demo_spf.jpg",
     "https://images.pexels.com/photos/5217958/pexels-photo-5217958.jpeg?auto=compress&cs=tinysrgb&w=600"),
    ("demo_lip_care.jpg",
     "https://images.pexels.com/photos/3373745/pexels-photo-3373745.jpeg?auto=compress&cs=tinysrgb&w=600"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def main():
    # Bypass SSL verification for DECIEM CDN (some macOS Python installs have cert issues)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    downloaded, failed = 0, 0

    for filename, url in DEMO_IMAGES:
        dest = IMAGES_DIR / filename
        if dest.exists():
            logger.info("  [skip] %s (already exists)", filename)
            downloaded += 1
            continue

        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                data = resp.read()
                if len(data) < 2000:
                    logger.warning("  [fail] %s (too small: %d bytes)", filename, len(data))
                    failed += 1
                    continue
                dest.write_bytes(data)
                logger.info("  [ok]   %s (%d KB)", filename, len(data) // 1024)
                downloaded += 1
        except Exception as e:
            logger.warning("  [fail] %s (%s)", filename, e)
            failed += 1

    logger.info("Done: %d downloaded, %d failed", downloaded, failed)
    logger.info("Images saved to: %s", IMAGES_DIR)


if __name__ == "__main__":
    main()
