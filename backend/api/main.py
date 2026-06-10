"""
FastAPI application for Pinterest Visual Search.

Endpoints
─────────
GET   /health            System status
GET   /products          List all indexed products
GET   /products/{id}     Single product detail
POST  /search/image      Visual similarity search (image upload)
POST  /search/text       Semantic text search
POST  /board/generate    Generate complementary routine board
"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import INDEX_DIR, DEFAULT_TOP_K, MAX_TOP_K, CORS_ORIGINS, DATA_DIR
from retrieval.clip_encoder import SigLIPEngine
from retrieval.faiss_index import VectorIndex
from agents.reranker import IngredientKB, rerank_for_complementarity, board_to_dict
from agents.narrator import generate_narrative
from agents.graph import run_board_agent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Global singletons (set during lifespan) ───────────────────────
encoder: SigLIPEngine | None = None
index: VectorIndex | None = None
catalog: dict = {}
ingredient_kb: IngredientKB | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, index, catalog, ingredient_kb

    logger.info("Starting up …")
    encoder = SigLIPEngine()

    index = VectorIndex()
    if (INDEX_DIR / "faiss.index").exists():
        index.load(INDEX_DIR)
        logger.info("FAISS index loaded (%d vectors)", index.size)
    else:
        logger.warning("No index found — run  python -m data.ingest  first")

    meta = INDEX_DIR / "products.json"
    if meta.exists():
        with open(meta) as f:
            catalog.update(json.load(f))
        logger.info("Product catalog loaded (%d entries)", len(catalog))

    kb_path = DATA_DIR / "ingredient_kb.json"
    if kb_path.exists():
        ingredient_kb = IngredientKB(kb_path)
    else:
        logger.warning("No ingredient KB found at %s", kb_path)

    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Pinterest Visual Search API",
    description="Multimodal skincare product discovery",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response schemas ──────────────────────────────────────────────

class ProductResult(BaseModel):
    id: str
    name: str
    brand: str
    category: str
    price: float
    description: str
    key_ingredients: list[str]
    skin_types: list[str]
    concerns: list[str]
    similarity_score: float


class SearchResponse(BaseModel):
    query_type: str
    results: list[ProductResult]
    total_indexed: int


# ── Helpers ───────────────────────────────────────────────────────

def _hydrate(results: list[tuple[str, float]]) -> list[ProductResult]:
    """Join FAISS results with product metadata."""
    out = []
    for pid, score in results:
        meta = catalog.get(pid)
        if meta is None:
            continue
        out.append(ProductResult(similarity_score=round(score, 4), **meta))
    return out


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_size": index.size if index else 0,
        "catalog_size": len(catalog),
    }


@app.get("/products")
def list_products():
    return {"products": list(catalog.values()), "total": len(catalog)}


@app.get("/products/{product_id}")
def get_product(product_id: str):
    if product_id not in catalog:
        raise HTTPException(404, f"Product '{product_id}' not found")
    return catalog[product_id]


@app.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    k: int = Query(default=DEFAULT_TOP_K, le=MAX_TOP_K, ge=1),
):
    """Upload an image → get visually similar products."""
    if not index or index.size == 0:
        raise HTTPException(503, "Index not loaded. Run ingestion first.")

    raw = await file.read()
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    emb = encoder.encode_image(img)
    hits = index.search(emb, k=k)
    return SearchResponse(
        query_type="image",
        results=_hydrate(hits),
        total_indexed=index.size,
    )


@app.post("/search/text", response_model=SearchResponse)
async def search_by_text(
    query: str = Query(..., min_length=1),
    k: int = Query(default=DEFAULT_TOP_K, le=MAX_TOP_K, ge=1),
):
    """Search by text description → get semantically similar products."""
    if not index or index.size == 0:
        raise HTTPException(503, "Index not loaded. Run ingestion first.")

    emb = encoder.encode_text(query)
    hits = index.search(emb, k=k)
    return SearchResponse(
        query_type="text",
        results=_hydrate(hits),
        total_indexed=index.size,
    )


@app.post("/board/generate")
async def generate_board(
    product_id: str = Query(..., description="ID of the anchor product"),
    k: int = Query(default=50, le=MAX_TOP_K, ge=10, description="FAISS retrieval depth"),
    board_size: int = Query(default=8, le=12, ge=4),
    narrate: bool = Query(default=False, description="Generate LLM narrative description"),
):
    """Generate a complementary skincare routine board.

    1. Retrieves top-k FAISS candidates by text similarity.
    2. Re-ranks candidates for routine complementarity.
    3. Optionally generates a Gemini-powered narrative description.
    4. Returns a curated board with coverage + conflict metrics.
    """
    if not index or index.size == 0:
        raise HTTPException(503, "Index not loaded.")
    if ingredient_kb is None:
        raise HTTPException(503, "Ingredient KB not loaded.")
    if product_id not in catalog:
        raise HTTPException(404, f"Product '{product_id}' not found")

    query_product = catalog[product_id]

    # Step 1: FAISS retrieval
    desc = f"{query_product['name']} {query_product['category']} {' '.join(query_product.get('concerns', []))}"
    emb = encoder.encode_text(desc)
    hits = index.search(emb, k=k)

    # Hydrate candidates
    candidates = []
    for pid, score in hits:
        meta = catalog.get(pid)
        if meta:
            candidates.append((meta, score))

    # Step 2: Complementarity re-ranking
    board = rerank_for_complementarity(
        query_product=query_product,
        candidates=candidates,
        ingredient_kb=ingredient_kb,
        max_board_size=board_size,
    )

    # Step 3: Serialize
    result = board_to_dict(query_product, board)

    # Step 4: Optional LLM narrative
    if narrate:
        narrative = generate_narrative(result)
        result["narrative"] = narrative

    return result


@app.post("/board/agent")
async def generate_board_agent(
    product_id: str = Query(..., description="ID of the anchor product"),
    k: int = Query(default=50, le=MAX_TOP_K, ge=10, description="FAISS retrieval depth"),
    board_size: int = Query(default=8, le=12, ge=4),
    narrate: bool = Query(default=False, description="Generate LLM narrative via Gemini"),
):
    """Generate a routine board via the LangGraph agent pipeline.

    Runs the full 4-node state machine:
      retrieve -> rerank -> narrate (optional) -> finalize

    Returns the same payload as /board/generate with an additional
    'agent_status' field confirming graph completion.
    """
    if product_id not in catalog:
        raise HTTPException(404, f"Product '{product_id}' not found")

    query_product = catalog[product_id]

    result = run_board_agent(
        product_id=product_id,
        query_product=query_product,
        k=k,
        board_size=board_size,
        narrate=narrate,
    )

    return result
