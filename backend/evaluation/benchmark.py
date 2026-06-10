"""
Benchmark runner for the board generation pipeline.

Samples products across all categories, generates boards via the live API,
computes aggregate metrics, and produces a formatted report.

Usage:
    cd backend
    python -m evaluation.benchmark              # default: 50 boards
    python -m evaluation.benchmark --n 100      # generate 100 boards
    python -m evaluation.benchmark --save       # save raw results to JSON

Requires the API server to be running on localhost:8000.
"""

import json
import logging
import random
import sys
import time
import urllib.request
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.evaluate import run_benchmark, routine_coverage, tier_consistency, avg_similarity

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8000"


def fetch_products() -> list[dict]:
    """Fetch the full product catalog from the API."""
    req = urllib.request.Request(f"{API_BASE}/products")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    return data["products"]


def generate_board(product_id: str, board_size: int = 8) -> dict | None:
    """Call the board/generate endpoint for a single product."""
    url = f"{API_BASE}/board/generate?product_id={product_id}&board_size={board_size}"
    req = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning("  Board generation failed for %s: %s", product_id, e)
        return None


def sample_products(products: list[dict], n: int) -> list[dict]:
    """Stratified sample: pick products proportionally from each category."""
    by_category: dict[str, list[dict]] = defaultdict(list)
    for p in products:
        by_category[p.get("category", "Unknown")].append(p)

    sampled = []
    categories = sorted(by_category.keys())
    per_cat = max(1, n // len(categories))
    remainder = n - per_cat * len(categories)

    for cat in categories:
        pool = by_category[cat]
        take = min(per_cat, len(pool))
        sampled.extend(random.sample(pool, take))

    # Fill remainder from largest categories
    if remainder > 0:
        remaining_pool = [p for p in products if p not in sampled]
        if remaining_pool:
            sampled.extend(random.sample(remaining_pool, min(remainder, len(remaining_pool))))

    return sampled[:n]


def print_report(results: dict, boards: list[dict], elapsed: float):
    """Print a formatted benchmark report."""
    sep = "-" * 60
    print()
    print(sep)
    print("  BENCHMARK REPORT")
    print(sep)
    print()
    print(f"  Boards generated:          {results['n_boards']}")
    print(f"  Time elapsed:              {elapsed:.1f}s ({elapsed/results['n_boards']:.2f}s per board)")
    print()
    print(f"  Avg routine coverage:      {results['avg_routine_coverage_pct']*100:.1f}%")
    print(f"  Perfect coverage rate:     {results['perfect_coverage_rate']*100:.1f}%")
    print(f"  Conflict-free rate:        {(1 - results['conflict_rate'])*100:.1f}%")
    print(f"  Avg complementarity score: {results['avg_complementarity_score']:.4f}")
    print(f"  Avg price std dev:         ${results['avg_tier_consistency']:.2f}")
    print()

    # Per-category breakdown
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for b in boards:
        cat = b.get("query_product", {}).get("category", "Unknown")
        by_cat[cat].append(b)

    print("  Category Breakdown:")
    print(f"  {'Category':<16} {'Count':>6} {'Coverage':>10} {'Conflict-Free':>15} {'Avg Score':>10}")
    print(f"  {'-'*14:<16} {'-'*5:>6} {'-'*8:>10} {'-'*13:>15} {'-'*9:>10}")

    for cat in sorted(by_cat.keys()):
        cat_boards = by_cat[cat]
        cat_results = run_benchmark(cat_boards)
        cf_rate = (1 - cat_results["conflict_rate"]) * 100
        print(
            f"  {cat:<16} {len(cat_boards):>6} "
            f"{cat_results['avg_routine_coverage_pct']*100:>9.1f}% "
            f"{cf_rate:>14.1f}% "
            f"{cat_results['avg_complementarity_score']:>10.4f}"
        )

    print()

    # Coverage distribution
    coverages = [routine_coverage(b)["covered"] for b in boards]
    print("  Coverage Distribution:")
    for steps in range(6):
        count = coverages.count(steps)
        bar = "#" * count
        print(f"    {steps}/5 steps: {count:>3} boards  {bar}")
    print()

    # Price range analysis
    price_ranges = []
    for b in boards:
        prices = [item.get("price", 0) for item in b.get("board", []) if item.get("price")]
        if prices:
            price_ranges.append((min(prices), max(prices)))

    if price_ranges:
        avg_min = sum(r[0] for r in price_ranges) / len(price_ranges)
        avg_max = sum(r[1] for r in price_ranges) / len(price_ranges)
        avg_spread = sum(r[1] - r[0] for r in price_ranges) / len(price_ranges)
        print(f"  Price Analysis:")
        print(f"    Avg min price:   ${avg_min:.2f}")
        print(f"    Avg max price:   ${avg_max:.2f}")
        print(f"    Avg spread:      ${avg_spread:.2f}")
        print()

    print(sep)


def main():
    n = 50
    save = "--save" in sys.argv
    for i, arg in enumerate(sys.argv):
        if arg == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])

    random.seed(42)  # reproducible sampling

    # Fetch catalog
    logger.info("Fetching product catalog from %s ...", API_BASE)
    products = fetch_products()
    logger.info("Catalog: %d products", len(products))

    # Sample
    sampled = sample_products(products, n)
    logger.info("Sampled %d products across %d categories", len(sampled), len(set(p["category"] for p in sampled)))

    # Generate boards
    boards = []
    t0 = time.time()
    for i, product in enumerate(sampled):
        pid = product["id"]
        board = generate_board(pid)
        if board:
            boards.append(board)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            logger.info(
                "  Progress: %d/%d boards generated (%.1fs elapsed)",
                len(boards), i + 1, elapsed,
            )

    elapsed = time.time() - t0
    logger.info("Generated %d boards in %.1fs", len(boards), elapsed)

    if not boards:
        logger.error("No boards generated. Is the API running?")
        sys.exit(1)

    # Compute metrics
    results = run_benchmark(boards)

    # Print report
    print_report(results, boards, elapsed)

    # Save raw results
    if save:
        out_path = Path(__file__).parent / "benchmark_results.json"
        with open(out_path, "w") as f:
            json.dump(
                {"config": {"n": n, "seed": 42}, "metrics": results, "boards": boards},
                f,
                indent=2,
            )
        logger.info("Raw results saved to %s", out_path)


if __name__ == "__main__":
    main()
