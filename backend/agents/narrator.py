"""
Gemini-powered narrative generator for skincare routine boards.

Takes the structured board output from the re-ranker and generates
natural language descriptions explaining product selection rationale,
ingredient complementarity, and routine flow.

Requires GEMINI_API_KEY environment variable.
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazy-init the Gemini client."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — narrative generation disabled")
        return None

    from google import genai
    _client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialized")
    return _client


SYSTEM_PROMPT = """You are a skincare formulation expert and routine consultant. Given a structured skincare routine board (JSON), generate a concise, expert narrative that:

1. Opens with a one-sentence summary of the routine's goal based on the anchor product.
2. For each product in the routine (in step order), write 1-2 sentences explaining:
   - Why this product was selected for this routine step
   - How its key ingredients complement or build on the previous step
   - Any notable ingredient synergies with other products in the board
3. Close with a brief note on the overall routine cohesion: price tier alignment, concern coverage, and any considerations for application order.

Rules:
- Be precise and technical but accessible. Reference specific ingredients by name.
- Do not use emojis.
- Do not recommend products outside the board.
- If the board has no ingredient conflicts, state that positively.
- Keep the total response under 300 words.
- Use plain paragraph format, no bullet points or headers."""


MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash",
]


def generate_narrative(board_data: dict[str, Any]) -> str | None:
    """Generate a narrative description for a board.

    Tries multiple Gemini models in fallback order to handle quota limits.

    Args:
        board_data: The full board dict from board_to_dict() including
                    query_product, board items, and metrics.

    Returns:
        Natural language narrative string, or None if unavailable.
    """
    client = _get_client()
    if client is None:
        return None

    # Build a compact JSON context for the LLM
    context = {
        "anchor": {
            "name": board_data["query_product"]["name"],
            "brand": board_data["query_product"]["brand"],
            "category": board_data["query_product"]["category"],
            "routine_step": board_data["query_product"].get("routine_step", ""),
        },
        "routine": [],
        "metrics": board_data.get("metrics", {}),
    }

    for item in board_data.get("board", []):
        context["routine"].append({
            "step": item.get("routine_step", ""),
            "name": item["name"],
            "brand": item["brand"],
            "price": item.get("price"),
            "key_ingredients": (
                item.get("key_ingredients", [])[:5]
                if isinstance(item.get("key_ingredients"), list)
                else []
            ),
            "concerns": item.get("concerns", []),
            "conflicts": item.get("ingredient_conflicts", []),
        })

    prompt = f"Routine board data:\n```json\n{json.dumps(context, indent=2)}\n```"

    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0.4,
                    "max_output_tokens": 500,
                },
            )
            narrative = response.text.strip()
            logger.info("Narrative generated via %s (%d chars)", model, len(narrative))
            return narrative
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                logger.warning("Model %s quota exhausted, trying next...", model)
                continue
            logger.error("Gemini narrative generation failed on %s: %s", model, e)
            return None

    logger.error("All Gemini models exhausted quota")
    return None
