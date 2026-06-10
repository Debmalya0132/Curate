"""
Complementarity Re-Ranker — the technical moat.

Given a query product and FAISS retrieval candidates, this module scores
candidates on ROUTINE FIT rather than visual similarity.

Scoring criteria:
  1. Routine step diversity  — does this fill a different step?
  2. Ingredient conflict-free — no retinol + AHA, etc.
  3. Price tier consistency   — similar price bracket as query
  4. Concern alignment        — targets the same skin concerns

The output is a curated "Pinterest Board" of 5-8 complementary products
that form a complete skincare routine.
"""

import json
import logging
import math
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Routine steps in canonical order ──────────────────────────────

ROUTINE_STEPS = ["cleanser", "toner", "serum", "treatment", "moisturizer", "spf"]

CATEGORY_TO_STEP = {
    "cleanser": "cleanser",
    "cleansing balm": "cleanser",
    "micellar water": "cleanser",
    "toner": "toner",
    "essence": "toner",
    "serum": "serum",
    "ampoule": "serum",
    "treatment": "treatment",
    "exfoliant": "treatment",
    "retinoid": "treatment",
    "spot treatment": "treatment",
    "moisturizer": "moisturizer",
    "cream": "moisturizer",
    "lotion": "moisturizer",
    "gel cream": "moisturizer",
    "face oil": "moisturizer",
    "sleeping mask": "moisturizer",
    "sunscreen": "spf",
    "spf": "spf",
    "eye cream": "eye cream",
    "lip care": "lip care",
    "mask": "mask",
}


# ── Ingredient KB ─────────────────────────────────────────────────

@dataclass
class Ingredient:
    name: str
    aliases: list[str]
    function: str
    routine_step: str
    conflicts: list[str]
    comedogenic_rating: int
    beneficial_for: list[str]


class IngredientKB:
    """In-memory knowledge base of skincare ingredients and their conflicts."""

    def __init__(self, kb_path: str | Path):
        self._ingredients: dict[str, Ingredient] = {}
        self._alias_map: dict[str, str] = {}
        self._load(kb_path)

    def _load(self, path: str | Path):
        with open(path) as f:
            raw = json.load(f)
        for entry in raw:
            ing = Ingredient(**entry)
            self._ingredients[ing.name] = ing
            for alias in ing.aliases:
                self._alias_map[alias.lower()] = ing.name
        logger.info("IngredientKB loaded: %d ingredients", len(self._ingredients))

    def resolve(self, name: str) -> str | None:
        """Resolve an ingredient name or alias to the canonical name."""
        low = name.lower().strip()
        if low in self._ingredients:
            return low
        return self._alias_map.get(low)

    def get(self, name: str) -> Ingredient | None:
        """Lookup ingredient info by name or alias."""
        canonical = self.resolve(name)
        if canonical:
            return self._ingredients[canonical]
        return None

    def find_conflicts(self, ingredients_a: list[str], ingredients_b: list[str]) -> list[tuple[str, str]]:
        """Find all ingredient conflicts between two product ingredient lists."""
        conflicts = []
        resolved_a = {self.resolve(i) for i in ingredients_a if self.resolve(i)}
        resolved_b = {self.resolve(i) for i in ingredients_b if self.resolve(i)}

        for name_a in resolved_a:
            ing = self._ingredients.get(name_a)
            if not ing:
                continue
            for conflict_name in ing.conflicts:
                if conflict_name in resolved_b:
                    conflicts.append((name_a, conflict_name))

        return conflicts


# ── Scoring ───────────────────────────────────────────────────────

def _classify_step(product: dict) -> str:
    """Map a product's category to a routine step."""
    cat = product.get("category", "").lower().strip()
    return CATEGORY_TO_STEP.get(cat, "other")


def _price_tier(price: float) -> str:
    """Bucket price into tiers for consistency scoring."""
    if price < 15:
        return "budget"
    elif price < 35:
        return "mid"
    elif price < 70:
        return "premium"
    else:
        return "luxury"


def _concern_overlap(query: dict, candidate: dict) -> float:
    """Score [0, 1] based on shared skin concerns."""
    q = set(c.lower() for c in query.get("concerns", []))
    c = set(c.lower() for c in candidate.get("concerns", []))
    if not q:
        return 0.5  # neutral
    return len(q & c) / len(q | c) if (q | c) else 0.0


@dataclass
class ScoredCandidate:
    product: dict
    routine_step: str
    similarity_score: float
    step_bonus: float = 0.0        # +bonus for filling an uncovered step
    conflict_penalty: float = 0.0  # -penalty for ingredient conflicts
    tier_bonus: float = 0.0        # +bonus for matching price tier
    concern_score: float = 0.0     # overlap with query concerns
    conflicts_found: list = field(default_factory=list)

    @property
    def total_score(self) -> float:
        return (
            self.similarity_score * 0.3
            + self.step_bonus * 0.3
            + self.concern_score * 0.2
            + self.tier_bonus * 0.1
            - self.conflict_penalty * 0.4
        )


def rerank_for_complementarity(
    query_product: dict,
    candidates: list[tuple[dict, float]],
    ingredient_kb: IngredientKB,
    max_board_size: int = 8,
) -> list[ScoredCandidate]:
    """
    Re-rank FAISS retrieval candidates for routine complementarity.

    Args:
        query_product: The uploaded/selected product's metadata.
        candidates: List of (product_metadata, similarity_score) from FAISS.
        ingredient_kb: The loaded ingredient knowledge base.
        max_board_size: Max products in the output board.

    Returns:
        Ordered list of ScoredCandidates forming a complementary routine.
    """
    query_step = _classify_step(query_product)
    query_tier = _price_tier(query_product.get("price", 0))
    query_ingredients = query_product.get("key_ingredients", [])

    # Score all candidates
    scored = []
    for product, sim_score in candidates:
        # Skip the query product itself
        if product.get("id") == query_product.get("id"):
            continue

        step = _classify_step(product)
        cand_tier = _price_tier(product.get("price", 0))
        cand_ingredients = product.get("key_ingredients", [])

        # Ingredient conflict check
        conflicts = ingredient_kb.find_conflicts(query_ingredients, cand_ingredients)
        conflict_penalty = min(1.0, len(conflicts) * 0.5)

        # Step diversity bonus (higher if different from query)
        step_bonus = 1.0 if step != query_step and step in ROUTINE_STEPS else 0.3

        # Price tier consistency
        tier_bonus = 1.0 if cand_tier == query_tier else 0.5

        # Concern overlap
        concern = _concern_overlap(query_product, product)

        sc = ScoredCandidate(
            product=product,
            routine_step=step,
            similarity_score=sim_score,
            step_bonus=step_bonus,
            conflict_penalty=conflict_penalty,
            tier_bonus=tier_bonus,
            concern_score=concern,
            conflicts_found=conflicts,
        )
        scored.append(sc)

    # Sort by total score descending
    scored.sort(key=lambda s: s.total_score, reverse=True)

    # Greedy selection: pick best candidate per routine step, then fill remaining
    board: list[ScoredCandidate] = []
    covered_steps: set[str] = set()

    # Phase 1: fill each routine step with the best conflict-free candidate
    for step in ROUTINE_STEPS:
        if step == query_step:
            continue  # skip the query's own step
        for sc in scored:
            if sc.routine_step == step and sc.product["id"] not in {b.product["id"] for b in board}:
                if not sc.conflicts_found:  # prefer conflict-free
                    board.append(sc)
                    covered_steps.add(step)
                    break

    # Phase 2: fill remaining slots with highest-scoring candidates
    for sc in scored:
        if len(board) >= max_board_size:
            break
        if sc.product["id"] not in {b.product["id"] for b in board}:
            board.append(sc)

    # Final sort: routine step order
    step_order = {s: i for i, s in enumerate(ROUTINE_STEPS)}
    board.sort(key=lambda s: step_order.get(s.routine_step, 99))

    logger.info(
        "Board generated: %d products, %d/%d routine steps covered, %d total conflicts",
        len(board),
        len(covered_steps),
        len(ROUTINE_STEPS) - 1,  # exclude query's step
        sum(len(b.conflicts_found) for b in board),
    )

    return board


# ── Board serialization ───────────────────────────────────────────

def board_to_dict(
    query_product: dict,
    board: list[ScoredCandidate],
) -> dict:
    """Serialize a generated board to a JSON-friendly dict."""
    covered = {sc.routine_step for sc in board if sc.routine_step in ROUTINE_STEPS}
    all_steps = set(ROUTINE_STEPS) - {_classify_step(query_product)}

    return {
        "query_product": {
            "id": query_product["id"],
            "name": query_product["name"],
            "brand": query_product["brand"],
            "category": query_product["category"],
            "routine_step": _classify_step(query_product),
        },
        "board": [
            {
                "id": sc.product["id"],
                "name": sc.product["name"],
                "brand": sc.product["brand"],
                "category": sc.product["category"],
                "price": sc.product["price"],
                "routine_step": sc.routine_step,
                "key_ingredients": sc.product.get("key_ingredients", []),
                "concerns": sc.product.get("concerns", []),
                "description": sc.product.get("description", ""),
                "complementarity_score": round(sc.total_score, 4),
                "ingredient_conflicts": [
                    {"ingredient_a": a, "ingredient_b": b}
                    for a, b in sc.conflicts_found
                ],
            }
            for sc in board
        ],
        "metrics": {
            "routine_coverage": len(covered),
            "routine_coverage_max": len(all_steps),
            "routine_coverage_pct": round(len(covered) / max(1, len(all_steps)), 3),
            "conflict_free": all(len(sc.conflicts_found) == 0 for sc in board),
            "total_conflicts": sum(len(sc.conflicts_found) for sc in board),
            "price_range": {
                "min": min((sc.product["price"] for sc in board), default=0),
                "max": max((sc.product["price"] for sc in board), default=0),
            },
        },
    }
