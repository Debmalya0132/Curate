# Multimodal Skincare Discovery

Agent-driven visual search and contextual recommender for skincare product discovery. Users upload an image or describe a product, and the system retrieves visually similar items from a real product corpus, then re-ranks the results for routine complementarity rather than raw similarity. The output is a curated board of products that form a complete skincare routine with zero ingredient conflicts.

This project targets the class of multimodal discovery problems that underpin visual search platforms: the gap between raw similarity retrieval and contextually useful recommendation. The technical contribution is the complementarity re-ranking layer, which transforms nearest-neighbor results into actionable, conflict-free routine construction.


## Table of Contents

- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Technical Stack](#technical-stack)
- [Dataset](#dataset)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Complementarity Re-Ranker](#complementarity-re-ranker)
- [Ingredient Knowledge Base](#ingredient-knowledge-base)
- [API Reference](#api-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Current Status](#current-status)
- [Roadmap](#roadmap)


## Architecture

The system follows a four-stage pipeline orchestrated by a LangGraph state machine:

```
Stage 1: Retrieval (SigLIP + FAISS)
    User input (image or text)
        -> SigLIP encoder produces a 768-dimensional normalized embedding
        -> FAISS IndexFlatIP performs inner-product nearest-neighbor search
        -> Returns top-50 candidate products with similarity scores

Stage 2: Re-Ranking (Complementarity Scorer)
    Top-50 FAISS candidates + query product metadata
        -> Ingredient conflict detection against the knowledge base
        -> Routine step classification (cleanser, toner, serum, treatment, moisturizer, SPF)
        -> Price tier consistency scoring
        -> Skin concern overlap calculation
        -> Greedy board assembly: best conflict-free candidate per routine step

Stage 3: Narrative Generation (Gemini 2.5 Flash)
    Structured board JSON
        -> Gemini LLM second pass via ?narrate=true
        -> Generates expert routine description: ingredient synergies, step rationale, cohesion
        -> Returns 200-300 word natural language narrative alongside the structured board

Stage 4: Agent Orchestration (LangGraph)
    BoardState typed state machine
        -> retrieve node: FAISS query, populates candidates
        -> rerank node: complementarity scoring, populates board
        -> narrate node: conditional Gemini call, populates narrative
        -> finalize node: serializes output, sets agent_status=done
        -> Conditional edges: routes to END on error at each stage
        -> Exposed via POST /board/agent endpoint
```

The key architectural insight is that similarity-based retrieval alone is insufficient for product recommendation. Returning a sunscreen that looks like the uploaded sunscreen is not useful. The complementarity re-ranker is what transforms raw retrieval into actionable routine construction, and the LangGraph layer is what makes the pipeline composable and auditable at each stage.


## Directory Structure

```
pinterest_visual_search_pro/
|
|-- backend/
|   |-- api/
|   |   |-- main.py                  # FastAPI application, 6 endpoints, CORS, lifespan startup
|   |
|   |-- agents/
|   |   |-- graph.py                 # LangGraph state machine for agent orchestration; wires the
|   |   |-- reranker.py              # Complementarity re-ranker, ingredient conflict detection,
|   |                                  routine step mapping, scoring, greedy board assembly
|   |
|   |-- retrieval/
|   |   |-- __init__.py              # Package exports: SigLIPEngine, VectorIndex
|   |   |-- clip_encoder.py          # SigLIP encoder with lazy model loading, batch encode,
|   |   |                              image and text encoding, L2 normalization
|   |   |-- faiss_index.py           # FAISS IndexFlatIP wrapper with add, search, save, load, reset
|   |
|   |-- data/
|   |   |-- seed_products.json       # Hand-curated 40-product skincare catalog (initial development)
|   |   |-- sephora/                 # Raw Sephora Kaggle CSV files (downloaded via kaggle CLI)
|   |   |   |-- product_info.csv     # 8,494 products with ingredients, categories, prices, ratings
|   |   |   |-- reviews_*.csv        # User review data (not currently used)
|   |   |-- sephora_products.json    # Parsed Sephora dataset: 3,413 skincare products in our schema
|   |   |-- ingredient_kb.json       # 65 skincare ingredients with function, conflicts, routine step,
|   |   |                              aliases, comedogenic rating, beneficial uses
|   |   |-- parse_sephora.py         # Sephora CSV parser: category mapping, ingredient extraction,
|   |   |                              highlights parsing, schema conversion
|   |   |-- ingest.py                # Ingestion pipeline: encodes products with SigLIP, builds FAISS
|   |   |                              index. Supports --sephora and --use-images flags
|   |   |-- images/                  # Product images directory (empty, for future image-based indexing)
|   |   |-- index/                   # Persisted FAISS index files (auto-generated)
|   |       |-- faiss.index          # 3,413 product vectors, 10 MB
|   |       |-- id_map.npy           # Product ID to FAISS index mapping
|   |       |-- products.json        # Full product metadata for API hydration
|   |
|   |-- evaluation/
|   |   |-- evaluate.py              # IR metrics (Recall@K, Precision@K, AP, nDCG, MRR) and
|   |                                  board metrics (routine coverage, conflict rate, tier consistency)
|   |
|   |-- core/
|   |   |-- __init__.py              # Shared utilities placeholder
|   |
|   |-- config.py                    # Central configuration: paths, model name, embedding dimension,
|   |                                  search defaults, API settings, CORS origins
|   |-- requirements.txt             # Pinned Python dependencies
|   |-- .venv/                       # Python virtual environment (not committed)
|
|-- frontend/
|   |-- package.json                 # Vite + React 18 project configuration
|   |-- vite.config.js               # Dev server with API proxy to backend :8000
|   |-- index.html                   # Entry point with Inter + Playfair Display fonts
|   |-- src/
|       |-- main.jsx                 # React entry point
|       |-- App.jsx                  # Main application: hero, search, results, board views
|       |-- api.js                   # API client (text search, image search, board generation)
|       |-- index.css                # Design system: ivory/cream palette, serif typography
|       |-- components/
|           |-- ProductCard.jsx       # Product card with brand, price, concern tags, score
|           |-- RoutineBoard.jsx      # Vertical timeline routine board with step indicators
|
|-- docker-compose.yml               # Multi-service container configuration
|-- .github/workflows/ci.yml         # CI pipeline scaffold
|-- render.yaml                      # Render Blueprint for one-click deployment
|-- README.md                        # This file
```


## Technical Stack

### Embedding Model: SigLIP

The system uses Google's SigLIP (Sigmoid Loss for Image-Language Pre-training) model, specifically `google/siglip-base-patch16-224`. This is a vision-language model that encodes both images and text into a shared 768-dimensional embedding space.

SigLIP was chosen over OpenAI's CLIP for two reasons: it is fully open-source with no API key required, and its sigmoid-based training objective produces better calibrated similarity scores for retrieval tasks compared to CLIP's contrastive softmax.

The encoder implementation in `clip_encoder.py` provides:

- Lazy model loading: the 813 MB model is downloaded from Hugging Face on first use, not on import.
- Device auto-detection: automatically uses CUDA, Apple MPS, or CPU depending on hardware.
- Batch encoding: `encode_images()` and `encode_texts()` accept lists for throughput.
- L2 normalization: all output vectors are unit-normalized so inner product equals cosine similarity.
- Compatibility layer: handles both raw tensor and `BaseModelOutputWithPooling` returns from different `transformers` library versions.

### Vector Search: FAISS

Facebook AI Similarity Search (FAISS) is used for nearest-neighbor retrieval. The index type is `IndexFlatIP` (flat inner product), which performs exact brute-force search. This is appropriate for the current corpus size of 3,413 vectors where exact search completes in under 1 millisecond.

The `faiss_index.py` implementation provides:

- `add(embeddings, ids)`: insert L2-normalized vectors with string product IDs.
- `search(query, k)`: return top-k product IDs with similarity scores.
- `save(directory)` / `load(directory)`: persist to disk as `faiss.index` + `id_map.npy`.
- `reset()`: clear the index for re-indexing.

The ID map is maintained as a parallel list that maps FAISS's internal integer indices to our string product IDs.

### API Framework: FastAPI

The API is built with FastAPI 0.115.0, using asynchronous request handling and Pydantic v2 for response validation. CORS middleware is configured for frontend development on localhost:5173 and localhost:3000.

Application startup uses FastAPI's lifespan context manager to load the FAISS index, product catalog, SigLIP model, and ingredient knowledge base once at startup rather than per-request.


## Dataset

### Source

The primary dataset is "Sephora Products and Skincare Reviews" from Kaggle, published under the CC BY 4.0 license. The raw download contains 8,494 products across all Sephora categories (makeup, fragrance, hair, etc.) with full ingredient lists, pricing, ratings, and category hierarchies.

### Parsing

The `parse_sephora.py` script filters and transforms the raw CSV into our schema:

1. Category filtering: only skincare-relevant categories are retained (cleansers, moisturizers, serums, sunscreens, toners, treatments, masks, lip care). Non-skincare products (makeup, fragrance, hair tools) are discarded.

2. Category normalization: Sephora's three-level category hierarchy (`primary_category`, `secondary_category`, `tertiary_category`) is mapped to eight simplified categories using keyword matching. For example, any product with "cleanser" or "face wash" in its category chain maps to "Cleanser".

3. Ingredient extraction: the raw ingredients field is a Python list literal stored as a string. The parser uses `ast.literal_eval` to parse it, then cleans each ingredient by removing parenthetical INCI names and truncating to the first 20 ingredients.

4. Highlights extraction: Sephora's highlights field contains tags like "Good for: Dryness", "Hydrating", "Clean at Sephora". These are parsed into skin type and concern lists.

After parsing, 3,413 skincare products remain with the following category distribution:

| Category    | Count |
|-------------|-------|
| Treatment   | 1,178 |
| Moisturizer |   738 |
| Cleanser    |   412 |
| Serum       |   379 |
| Lip Care    |   328 |
| Mask        |   180 |
| Sunscreen   |   111 |
| Toner       |    87 |

### Indexing

The ingestion pipeline (`ingest.py`) supports two modes:

**Text mode** (default for full corpus): Each product is converted into a rich text description combining its name, brand, category, skin types, concerns, and top 5 ingredients. These descriptions are encoded with SigLIP's text encoder in batches of 16. This mode requires no product images and produces a functional index in approximately 30 seconds on an Apple M-series chip.

**Image mode** (available for subset indexing): Looks for product images at `data/images/<product_id>.jpg` and encodes them with SigLIP's image encoder. This mode produces higher-fidelity visual embeddings and is used when product images are available.

**Cross-modal retrieval**: Because SigLIP maps both images and text into the same 768-dimensional joint embedding space, image queries work against the text-mode index and vice versa. This is a fundamental property of contrastive vision-language pre-training. Similarity scores for cross-modal queries are lower than same-modal queries (approximately 0.05-0.10 for image-to-text versus 0.40-0.55 for text-to-text), which is expected and well-documented in the CLIP/SigLIP literature. Cross-modal retrieval still returns category-correct results (a photo of a moisturizer returns moisturizers) because the joint embedding captures semantic similarity across modalities.

The generated FAISS index is approximately 10 MB on disk and contains 3,413 768-dimensional float32 vectors.

### Seed Catalog

A secondary hand-curated catalog of 40 products (`seed_products.json`) was created during initial development for rapid iteration before the Sephora dataset was integrated. It contains well-known skincare products from CeraVe, The Ordinary, La Roche-Posay, Drunk Elephant, SkinCeuticals, and other brands, with manually written descriptions, ingredient lists, and concern tags. This catalog can still be used by running ingestion without the `--sephora` flag.


## Retrieval Pipeline

The retrieval pipeline is the first stage of the system. When a user submits a query (image upload or text description):

1. The SigLIP encoder converts the input into a 768-dimensional unit-normalized vector.
2. FAISS performs inner-product search against all 3,413 indexed product vectors.
3. The top-k results (default k=12, max k=50) are returned with cosine similarity scores.
4. Each result is hydrated with full product metadata from the in-memory catalog.

For text queries, the pipeline achieves similarity scores in the 0.40-0.55 range, which is expected for text-to-text similarity in SigLIP's embedding space (the model is primarily trained on image-text pairs). Image queries will produce higher scores when image-based indexing is implemented.


## Complementarity Re-Ranker

The re-ranker (`agents/reranker.py`) is the core technical contribution of this project. It transforms similarity-based retrieval results into a routine-aware recommendation board.

### Scoring Dimensions

Each FAISS candidate is scored across four dimensions:

1. **Routine step diversity** (weight: 0.3). Products are classified into routine steps: cleanser, toner, serum, treatment, moisturizer, SPF. Candidates that fill a different routine step than the query product receive a bonus of 1.0; candidates in the same step receive 0.3.

2. **Ingredient conflict penalty** (weight: 0.4). The re-ranker checks every candidate's ingredients against the query product's ingredients using the ingredient knowledge base. Known conflicts (e.g., retinol and salicylic acid, vitamin C and benzoyl peroxide) incur a penalty of 0.5 per conflict, capped at 1.0.

3. **Concern overlap** (weight: 0.2). Candidates that target the same skin concerns as the query product (e.g., both targeting "hydration" and "anti-aging") receive a higher score. Computed as the Jaccard similarity of the concern sets.

4. **Price tier consistency** (weight: 0.1). Products are bucketed into four tiers: budget (under $15), mid ($15-$35), premium ($35-$70), luxury (over $70). Matching the query product's tier scores 1.0; mismatches score 0.5.

The total complementarity score is:

```
total = similarity * 0.3 + step_bonus * 0.3 + concern * 0.2 + tier * 0.1 - conflict_penalty * 0.4
```

### Board Assembly

After scoring, the re-ranker uses a two-phase greedy algorithm:

Phase 1: For each of the 5 routine steps (excluding the query product's own step), select the highest-scoring conflict-free candidate. This ensures the board covers as many routine steps as possible.

Phase 2: Fill remaining board slots with the highest-scoring candidates regardless of step, up to the maximum board size.

The final board is sorted by canonical routine step order: cleanser, toner, serum, treatment, moisturizer, SPF.

### Category-to-Step Mapping

The re-ranker maintains a mapping table from product categories to routine steps. For example:

- "Cleanser", "Cleansing Balm", "Micellar Water" all map to the "cleanser" step.
- "Serum", "Ampoule" map to "serum".
- "Sunscreen", "SPF" map to "spf".
- "Face Oil", "Sleeping Mask" map to "moisturizer".
- "Eye Cream" and "Lip Care" map to their own steps outside the core routine.


## Ingredient Knowledge Base

The ingredient knowledge base (`data/ingredient_kb.json`) contains 65 common skincare ingredients with structured metadata. Each entry includes:

| Field              | Type         | Description                                                        |
|--------------------|--------------|--------------------------------------------------------------------|
| name               | string       | Canonical ingredient name (e.g., "retinol")                        |
| aliases            | list[string] | Alternative names (e.g., ["vitamin a", "retinoid", "retinal"])     |
| function           | string       | What the ingredient does (e.g., "cell turnover accelerator")       |
| routine_step       | string       | Which routine step this ingredient typically belongs to             |
| conflicts          | list[string] | Ingredients that should not be combined with this one               |
| comedogenic_rating | int          | Pore-clogging risk on a 0-5 scale                                  |
| beneficial_for     | list[string] | Skin concerns this ingredient addresses                            |

### Key Conflict Pairs

The knowledge base encodes the following clinically recognized ingredient conflicts:

- Retinol conflicts with: salicylic acid, glycolic acid, benzoyl peroxide, vitamin C, lactic acid, mandelic acid
- Vitamin C conflicts with: retinol, benzoyl peroxide, copper peptides
- Glycolic acid conflicts with: retinol, salicylic acid, vitamin C, benzoyl peroxide
- Benzoyl peroxide conflicts with: retinol, vitamin C, glycolic acid, salicylic acid, lactic acid
- Copper peptides conflict with: vitamin C, retinol, glycolic acid

### Resolution

The `IngredientKB` class supports alias resolution, so "vitamin a" resolves to "retinol", "BHA" resolves to "salicylic acid", and "l-ascorbic acid" resolves to "vitamin c". This ensures that conflict detection works regardless of how ingredients are named in different product listings.


## API Reference

The API runs on port 8000 and serves 6 endpoints. Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### GET /health

Returns system status including index size and catalog size.

Response:
```json
{
  "status": "ok",
  "index_size": 3413,
  "catalog_size": 3413
}
```

### GET /products

Returns the full product catalog.

Response:
```json
{
  "products": [ ... ],
  "total": 3413
}
```

### GET /products/{product_id}

Returns a single product by ID. Returns 404 if not found.

### POST /search/image

Accepts a multipart file upload. Encodes the uploaded image with SigLIP, searches FAISS, and returns the top-k most visually similar products.

Query parameters:
- `k` (int, default=12, max=50): number of results to return

Response:
```json
{
  "query_type": "image",
  "results": [
    {
      "id": "P442830",
      "name": "100% Plant-Derived Squalane",
      "brand": "The Ordinary",
      "category": "Serum",
      "price": 10.0,
      "description": "...",
      "key_ingredients": ["..."],
      "skin_types": ["..."],
      "concerns": ["..."],
      "similarity_score": 0.5234
    }
  ],
  "total_indexed": 3413
}
```

### POST /search/text

Encodes a text query with SigLIP and returns semantically similar products.

Query parameters:
- `query` (string, required): search text
- `k` (int, default=12, max=50): number of results

### POST /board/generate

The primary endpoint. Given an anchor product ID, retrieves FAISS candidates, re-ranks them for routine complementarity, and returns a curated board.

Query parameters:
- `product_id` (string, required): anchor product ID
- `k` (int, default=50): FAISS retrieval depth
- `board_size` (int, default=8, range 4-12): max products in the board

Response:
```json
{
  "query_product": {
    "id": "P476416",
    "name": "AFRICAN Beauty Butter",
    "brand": "54 Thrones",
    "category": "Moisturizer",
    "routine_step": "moisturizer"
  },
  "board": [
    {
      "id": "P476415",
      "name": "Acne+ 2% BHA Cleanser",
      "brand": "Skinfix",
      "category": "Cleanser",
      "price": 35.0,
      "routine_step": "cleanser",
      "key_ingredients": ["..."],
      "concerns": ["..."],
      "complementarity_score": 0.5473,
      "ingredient_conflicts": []
    }
  ],
  "metrics": {
    "routine_coverage": 4,
    "routine_coverage_max": 5,
    "routine_coverage_pct": 0.8,
    "conflict_free": true,
    "total_conflicts": 0,
    "price_range": { "min": 7.0, "max": 38.0 }
  }
}
```


## Evaluation Metrics

The evaluation module (`evaluation/evaluate.py`) provides two categories of metrics.

### Standard IR Metrics

These measure retrieval quality against a ground truth set:

- **Recall@K**: fraction of relevant items found in the top-k results.
- **Precision@K**: fraction of top-k results that are relevant.
- **Average Precision**: average of precision values computed at each relevant position.
- **nDCG@K**: normalized discounted cumulative gain, accounts for ranking position.
- **MRR**: mean reciprocal rank, 1 divided by the rank of the first relevant result.

### Routine-Aware Board Metrics

These measure the quality of generated boards:

- **Routine Coverage**: how many of the 5 standard routine steps are covered by the board (0-5 integer, also expressed as a percentage).
- **Conflict Rate**: fraction of generated boards that contain at least one known ingredient conflict. Computed programmatically, not estimated.
- **Tier Consistency**: standard deviation of prices across board items. Lower values indicate a more cohesive price tier.
- **Average Complementarity Score**: mean of the re-ranker's composite scores across board items.
- **Perfect Coverage Rate**: fraction of boards that achieve 100% routine step coverage.

### Benchmark Results

The benchmark runner (`evaluation/benchmark.py`) samples products stratified across all categories, generates boards via the live API, and reports aggregate metrics. Results below are from a 50-board run with `random.seed(42)` across 8 product categories.

| Metric | Result |
|--------|--------|
| Boards evaluated | 50 |
| Average routine coverage | **83.3%** (4.2 / 5 steps) |
| Perfect 5/5 coverage rate | **38%** of boards |
| Conflict-free rate | **100%** across all 50 boards |
| Average complementarity score | 0.6119 |
| Latency | 102ms per board (50 boards in 5.1s) |
| Boards with zero-step coverage | 0 |

Coverage distribution: every board covers at least 4 of 5 standard routine steps. No board was generated with fewer than 4 steps covered.

Per-category results:

| Category | Boards | Avg Coverage | Conflict-Free | Avg Score |
|----------|--------|-------------|---------------|-----------|
| Cleanser | 6 | 80.0% | 100% | 0.6184 |
| Lip Care | 6 | 83.3% | 100% | 0.6475 |
| Mask | 7 | 83.3% | 100% | 0.5882 |
| Moisturizer | 6 | 80.0% | 100% | 0.5917 |
| Serum | 6 | 80.0% | 100% | 0.5794 |
| Sunscreen | 6 | **100.0%** | 100% | 0.6511 |
| Toner | 6 | 80.0% | 100% | 0.6192 |
| Treatment | 7 | 80.0% | 100% | 0.6046 |

Sunscreen-anchored boards achieve 100% coverage because SPF is the terminal routine step, so the re-ranker fills all prior steps by definition.

To reproduce:

```bash
cd backend
python3 -m evaluation.benchmark --n 50 --save
```


## Setup and Installation

### Prerequisites

- Python 3.10 or higher (developed and tested on Python 3.14)
- macOS, Linux, or Windows
- Approximately 1 GB of free disk space for the SigLIP model download
- Kaggle account and API token (for downloading the Sephora dataset)

### Step 1: Clone and Navigate

```bash
git clone <repository-url>
cd pinterest_visual_search_pro
```

### Step 2: Create Virtual Environment and Install Dependencies

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The installation will download approximately 500 MB of Python packages including PyTorch, Transformers, FAISS, and FastAPI.

### Step 3: Download the Sephora Dataset

This requires a Kaggle API token. Place your `kaggle.json` file at `~/.kaggle/kaggle.json`.

```bash
pip install kaggle
kaggle datasets download -d nadyinky/sephora-products-and-skincare-reviews -p data/sephora --unzip
```

### Step 4: Parse the Dataset

```bash
python3 -m data.parse_sephora --stats
```

This filters the 8,494 raw products down to 3,413 skincare products and writes `data/sephora_products.json`.

### Step 5: Build the FAISS Index

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 -m data.ingest --sephora
```

On the first run, this downloads the SigLIP model from Hugging Face (approximately 813 MB). Subsequent runs use the cached model. The ingestion encodes all 3,413 products in approximately 30 seconds on an Apple M-series chip and writes the index to `data/index/`.

The `KMP_DUPLICATE_LIB_OK=TRUE` environment variable is required on macOS to resolve an OpenMP library conflict between PyTorch and FAISS. It is not needed on Linux.

To use the small 40-product seed catalog instead (for faster iteration):

```bash
python3 -m data.ingest
```

### Step 6: Start the API Server

```bash
KMP_DUPLICATE_LIB_OK=TRUE uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The server loads the FAISS index, product catalog, and ingredient knowledge base at startup. The SigLIP model is lazy-loaded on the first search request.

### Step 7: Verify

```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","index_size":3413,"catalog_size":3413}

curl -X POST "http://localhost:8000/search/text?query=vitamin+c+serum+brightening&k=3"
# Returns top-3 semantically similar products

curl -X POST "http://localhost:8000/board/generate?product_id=P476416&board_size=6"
# Returns a complementary routine board with coverage and conflict metrics
```

Interactive API documentation is available at `http://localhost:8000/docs`.


## Usage

### Text Search

```bash
curl -X POST "http://localhost:8000/search/text?query=gentle+cleanser+for+sensitive+skin&k=5"
```

### Image Search

```bash
curl -X POST -F "file=@sunscreen.jpg" "http://localhost:8000/search/image?k=10"
```

### Board Generation

```bash
# Generate a routine board starting from a specific product
curl -X POST "http://localhost:8000/board/generate?product_id=P476416&board_size=8"
```

The board endpoint performs the full three-stage pipeline: FAISS retrieval of 50 candidates, complementarity re-ranking with ingredient conflict detection, and greedy board assembly across routine steps.


## Current Status

### Implemented

- SigLIP encoder with lazy loading, batch processing, image and text encoding, L2 normalization, and transformers v5 compatibility.
- FAISS vector index with add, search, save, load, and reset operations.
- Sephora Kaggle dataset integration: download, parsing, category mapping, ingredient extraction. 3,413 skincare products indexed.
- Hand-curated seed catalog of 40 products for rapid development.
- Ingredient knowledge base with 65 ingredients, alias resolution, and conflict mapping.
- Complementarity re-ranker with 4-dimensional scoring, ingredient conflict detection, and greedy board assembly.
- FastAPI application with 6 endpoints: health, products, product detail, image search, text search, and board generation.
- Evaluation module with standard IR metrics and routine-aware board metrics.
- Full ingestion pipeline supporting text-mode and image-mode encoding, with Sephora and seed catalog data sources.
- Cross-modal retrieval validated: image queries against text-indexed FAISS return category-correct results.
- React/Vite frontend with hero search, product result grid, drag-and-drop image upload, and vertical-timeline routine board view.
- LangGraph 1.2.4 state machine wired and running: retrieve -> rerank -> narrate -> finalize with conditional error routing.
- Gemini 2.5 Flash narrative generation wired to `/board/generate?narrate=true` and `/board/agent?narrate=true`.
- Docker multi-stage builds for backend (Python 3.11-slim) and frontend (Node 20 + nginx:alpine).
- docker-compose.yml for local containerized development.
- render.yaml Blueprint spec for one-click Render deployment.

### Known Limitations

- The full 3,413-product FAISS index uses text-mode embeddings. Cross-modal image queries work but produce lower similarity scores than same-modal queries (0.05-0.10 vs 0.40-0.55). Image-mode indexing on a product image subset is supported by the pipeline and documented in the Indexing section.
- Docker builds require Docker Desktop to be running locally. The Dockerfiles are validated and the frontend production build (`npm run build`) completes successfully.
- Image-mode indexing at full corpus scale is pending a product image acquisition pipeline.


## Deployment

### Local (Docker Compose)

Requires Docker Desktop to be running.

```bash
# Copy your Gemini API key to the root .env
echo "GEMINI_API_KEY=your_key_here" > .env

# Build and start both services
docker compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:80
```

### Render (Production)

The `render.yaml` Blueprint file enables one-click deployment.

1. Push the repository to GitHub.
2. Go to `dashboard.render.com` → New → Blueprint.
3. Connect the repository. Render reads `render.yaml` automatically.
4. Set `GEMINI_API_KEY` as an environment variable in the Render dashboard for the `curate-backend` service.
5. Deploy. Both services will build and start automatically.

The backend `curate-backend` service exposes the API at `https://curate-backend.onrender.com` and the frontend `curate-frontend` proxies all API calls to it via its nginx configuration.

Note: Render's free tier spins down services after 15 minutes of inactivity. The first request after spin-down will take 30-60 seconds while the SigLIP model loads into memory.


## Roadmap

### Phase 5: Observability and Performance

Add structured logging with request tracing across the LangGraph nodes, Prometheus metrics endpoint, and Redis caching for repeat product lookups. Image-mode indexing at full corpus scale as a parallel pipeline with a dedicated worker service.
