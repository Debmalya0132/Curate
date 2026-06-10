"""
LangGraph state machine for the skincare discovery agent.

The pipeline is modelled as a directed acyclic graph with four nodes:

    retrieve --> rerank --> narrate --> finalize

State transitions:
    retrieve  : FAISS vector search; populates candidates
    rerank    : complementarity re-ranking; populates board
    narrate   : Gemini LLM second pass; populates narrative (optional)
    finalize  : serializes output; sets status to 'done'

The graph is compiled once at import time and reused across requests.
Invoke it with:

    result = agent_graph.invoke(initial_state)

where initial_state is a BoardState dict.
"""

import logging
from typing import Any, Annotated
import operator

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ── State schema ──────────────────────────────────────────────────
# Each field is accumulated across nodes. list fields use operator.add
# so nodes can append without overwriting prior state.

class BoardState(TypedDict):
    # Input
    product_id: str
    query_product: dict[str, Any]
    k: int                                        # FAISS retrieval depth
    board_size: int
    narrate: bool

    # Node outputs (accumulated)
    candidates: Annotated[list[tuple[dict, float]], operator.add]
    board: list[dict[str, Any]]
    narrative: str | None

    # Control
    status: str                                   # 'pending' | 'retrieved' | 'reranked' | 'done'
    error: str | None


# ── Nodes ─────────────────────────────────────────────────────────

def retrieve_node(state: BoardState) -> dict[str, Any]:
    """
    Node 1: FAISS vector retrieval.

    Encodes the anchor product description with SigLIP and queries the
    FAISS index for the top-k nearest neighbours.
    """
    from retrieval.clip_encoder import SigLIPEngine
    from retrieval.faiss_index import VectorIndex
    from config import INDEX_DIR, DATA_DIR
    import json

    logger.info("[retrieve] product_id=%s k=%d", state["product_id"], state["k"])

    try:
        # Load shared singletons (they cache internally after first load)
        encoder = SigLIPEngine()

        index = VectorIndex()
        if (INDEX_DIR / "faiss.index").exists():
            index.load(INDEX_DIR)

        catalog_path = INDEX_DIR / "products.json"
        catalog: dict = {}
        if catalog_path.exists():
            with open(catalog_path) as f:
                catalog = json.load(f)

        qp = state["query_product"]
        desc = f"{qp['name']} {qp['category']} {' '.join(qp.get('concerns', []))}"
        emb = encoder.encode_text(desc)
        hits = index.search(emb, k=state["k"])

        candidates = []
        for pid, score in hits:
            meta = catalog.get(pid)
            if meta:
                candidates.append((meta, score))

        logger.info("[retrieve] %d candidates found", len(candidates))
        return {"candidates": candidates, "status": "retrieved", "error": None}

    except Exception as e:
        logger.error("[retrieve] failed: %s", e)
        return {"candidates": [], "status": "error", "error": str(e)}


def rerank_node(state: BoardState) -> dict[str, Any]:
    """
    Node 2: Complementarity re-ranking.

    Scores candidates on routine step diversity, ingredient conflict
    avoidance, price tier consistency, and concern overlap. Assembles
    a greedy board up to board_size products.
    """
    from agents.reranker import IngredientKB, rerank_for_complementarity
    from config import DATA_DIR

    logger.info("[rerank] %d candidates → board_size=%d", len(state["candidates"]), state["board_size"])

    if state.get("status") == "error" or not state["candidates"]:
        return {"board": [], "status": "error", "error": state.get("error", "No candidates")}

    try:
        kb_path = DATA_DIR / "ingredient_kb.json"
        ingredient_kb = IngredientKB(kb_path)

        board = rerank_for_complementarity(
            query_product=state["query_product"],
            candidates=state["candidates"],
            ingredient_kb=ingredient_kb,
            max_board_size=state["board_size"],
        )

        logger.info("[rerank] board assembled: %d products", len(board))
        return {"board": board, "status": "reranked", "error": None}

    except Exception as e:
        logger.error("[rerank] failed: %s", e)
        return {"board": [], "status": "error", "error": str(e)}


def narrate_node(state: BoardState) -> dict[str, Any]:
    """
    Node 3: Gemini LLM narrative generation (conditional).

    Generates a natural language routine description if narrate=True.
    Gracefully skips and returns narrative=None if the key is missing
    or the API call fails.
    """
    if not state.get("narrate", False):
        logger.info("[narrate] skipped (narrate=False)")
        return {"narrative": None}

    if not state["board"]:
        logger.info("[narrate] skipped (empty board)")
        return {"narrative": None}

    logger.info("[narrate] generating narrative via Gemini")

    try:
        from agents.reranker import board_to_dict
        from agents.narrator import generate_narrative

        board_data = board_to_dict(state["query_product"], state["board"])
        narrative = generate_narrative(board_data)
        logger.info("[narrate] done (%d chars)", len(narrative) if narrative else 0)
        return {"narrative": narrative}

    except Exception as e:
        logger.error("[narrate] failed: %s", e)
        return {"narrative": None}


def finalize_node(state: BoardState) -> dict[str, Any]:
    """
    Node 4: Finalize output.

    Attaches narrative to the board dict and sets status to 'done'.
    This node is the terminal sink — its output is the graph result.
    """
    from agents.reranker import board_to_dict

    logger.info("[finalize] board=%d products narrative=%s",
                len(state.get("board", [])),
                "yes" if state.get("narrative") else "no")

    result = board_to_dict(state["query_product"], state.get("board", []))
    if state.get("narrative"):
        result["narrative"] = state["narrative"]

    return {"status": "done"}


# ── Routing ───────────────────────────────────────────────────────

def route_after_retrieve(state: BoardState) -> str:
    """Route to rerank on success, END on error."""
    return "rerank" if state.get("status") != "error" else END


def route_after_rerank(state: BoardState) -> str:
    """Route to narrate on success, END on error."""
    return "narrate" if state.get("status") != "error" else END


# ── Graph compilation ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(BoardState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("narrate", narrate_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("retrieve")

    graph.add_conditional_edges("retrieve", route_after_retrieve, {"rerank": "rerank", END: END})
    graph.add_conditional_edges("rerank", route_after_rerank, {"narrate": "narrate", END: END})
    graph.add_edge("narrate", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# Compiled graph — import and invoke this
agent_graph = build_graph()


# ── Convenience runner ────────────────────────────────────────────

def run_board_agent(
    product_id: str,
    query_product: dict[str, Any],
    k: int = 50,
    board_size: int = 8,
    narrate: bool = False,
) -> dict[str, Any]:
    """
    Run the full board generation agent graph.

    Returns the board_to_dict payload augmented with 'narrative' if narrate=True.
    """
    initial_state: BoardState = {
        "product_id": product_id,
        "query_product": query_product,
        "k": k,
        "board_size": board_size,
        "narrate": narrate,
        "candidates": [],
        "board": [],
        "narrative": None,
        "status": "pending",
        "error": None,
    }

    final_state = agent_graph.invoke(initial_state)

    # Reconstruct result from final state
    from agents.reranker import board_to_dict
    result = board_to_dict(query_product, final_state.get("board", []))
    if final_state.get("narrative"):
        result["narrative"] = final_state["narrative"]
    result["agent_status"] = final_state.get("status", "unknown")
    return result
