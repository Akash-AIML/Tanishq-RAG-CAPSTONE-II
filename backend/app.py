# ============================================================
# JEWELLERY MULTIMODAL SEARCH BACKEND (FASTAPI) - COMPLETE FIXED VERSION
# ============================================================

# %%
# ============================================================
# IMPORTS
# ============================================================

import os
import json
from typing import List, Dict

import torch
import clip
import numpy as np
import chromadb

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

import base64
import requests
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()

# %%
# ============================================================
# CONFIG
# ============================================================

# Auto-detect if running in Docker (HF Spaces) or locally
if os.path.exists("/app/data"):
    # Running in Docker (Hugging Face Spaces)
    BASE_DIR = "/app"
else:
    # Running locally
    BASE_DIR = "/home/akash/Jewellary_RAG"

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_primary")
DATA_DIR = os.path.join(BASE_DIR, "data", "tanishq")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
EXTRACTED_METADATA_PATH = os.path.join(DATA_DIR, "enhanced_metadata.json")
FLORENCE_CAPTIONS_PATH = os.path.join(DATA_DIR, "florence_captions_all.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# ============================================================
# LAZY MODEL LOADING (Reduces cold start time)
# ============================================================

# Global model references (loaded on first use)
clip_model = None
cross_encoder = None

def get_clip_model():
    """Lazy load CLIP model on first use"""
    global clip_model
    if clip_model is None:
        print("🔹 Loading CLIP model...")
        model, _ = clip.load("ViT-B/16", device=DEVICE)
        model.eval()
        clip_model = model
        print("✅ CLIP model loaded")
    return clip_model

def get_cross_encoder():
    """Lazy load Cross-Encoder on first use"""
    global cross_encoder
    if cross_encoder is None:
        print("🔹 Loading Cross-Encoder for re-ranking...")
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✅ Cross-Encoder loaded")
    return cross_encoder

# %%
print("🔹 Loading enhanced metadata...")
with open(EXTRACTED_METADATA_PATH, "r") as f:
    ENHANCED_METADATA = json.load(f)

print("🔹 Loading Florence captions...")
with open(FLORENCE_CAPTIONS_PATH, "r") as f:
    florence_data = json.load(f)
    # Convert list format to dict for faster lookup
    FLORENCE_CAPTIONS = {item["image"]: item["caption"] for item in florence_data}

print(f"✅ Loaded metadata for {len(ENHANCED_METADATA)} images")
print(f"✅ Loaded {len(FLORENCE_CAPTIONS)} Florence captions")

# %%
# ============================================================
# INITIALIZE GROQ LLM CLIENT
# ============================================================

print("🔹 Initializing Groq LLM client...")

# Primary and fallback API keys
GROQ_API_KEYS = [
    os.environ.get("GROQ_API_KEY"),
    os.environ.get("GROQ_API_KEY_2"),  # Fallback key
    os.environ.get("GROQ_API_KEY_3"),  # Second fallback
]
# Filter out None values
GROQ_API_KEYS = [key for key in GROQ_API_KEYS if key]

if not GROQ_API_KEYS:
    print("⚠️ No Groq API keys found in environment variables")
    groq_client = None
else:
    print(f"✅ Loaded {len(GROQ_API_KEYS)} Groq API key(s)")
    # Initialize with primary key
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEYS[0]
    )

current_groq_key_index = 0

# NVIDIA OCR API configuration
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_OCR_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1"

# Fallback OCR configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# %%
# ============================================================
# LOAD CHROMA (PERSISTED DB)
# ============================================================

print("🔹 Connecting to Chroma DB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

image_collection = chroma_client.get_collection("jewelry_images")
metadata_collection = chroma_client.get_collection("jewelry_metadata")

print(
    "✅ Chroma loaded | Images:",
    image_collection.count(),
    "| Metadata:",
    metadata_collection.count()
)

# %%
# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Jewellery Multimodal Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://tanishq-rag-capstone-1lj2x4y1v-akash-aimls-projects.vercel.app",  # Vercel preview
        "https://*.vercel.app",  # All Vercel deployments
        "https://*.ngrok-free.app",  # Allow ngrok tunnels
        "https://*.ngrok-free.dev",  # Allow ngrok tunnels (new domain)
        "https://*.ngrok.io",         # Allow ngrok tunnels (legacy)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# %%
# ============================================================
# MIDDLEWARE FOR HF SPACES OPTIMIZATION
# ============================================================

import asyncio
from starlette.requests import Request

@app.middleware("http")
async def add_optimizations(request: Request, call_next):
    """Add upload size limits and request timeouts"""
    
    # Limit upload size to 5MB
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 5 * 1024 * 1024:  # 5MB
            raise HTTPException(status_code=413, detail="File too large (max 5MB)")
    
    # Add request timeout (45s for complex OCR queries)
    try:
        response = await asyncio.wait_for(call_next(request), timeout=45.0)
        return response
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout (max 45s)")

# %%
# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

from typing import Optional

class TextSearchRequest(BaseModel):
    query: str
    
    # Dynamic retrieval parameters
    top_k: int = 12  # Final results to return
    k_per_query: int = 12  # Retrieval depth per expanded query
    max_candidates: int = 40  # Pre-reranking candidate limit
    
    # Pagination
    page: int = 1
    page_size: int = 12
    
    # Performance toggles
    use_reranking: bool = True  # Toggle cross-encoder (3x faster when False)
    use_explanations: bool = True  # Toggle LLM explanations (500ms+ faster when False)
    
    # Inclusion filters
    category: Optional[str] = None  # "ring", "necklace", "earring", "bracelet"
    metal: Optional[str] = None  # "gold", "silver", "platinum"
    primary_stone: Optional[str] = None  # "diamond", "pearl", "ruby", etc.
    form: Optional[str] = None  # "plain", "studded"
    
    # Exclusion filters
    exclude_category: Optional[str] = None
    exclude_metal: Optional[str] = None
    exclude_stone: Optional[str] = None
    
    # Confidence thresholds (0.0-1.0)
    confidence_category: Optional[float] = None
    confidence_metal: Optional[float] = None
    confidence_primary_stone: Optional[float] = None
    
    # Similarity mode
    similarity_mode: str = "balanced"  # "strict", "balanced", "exploratory"


class SimilarSearchRequest(BaseModel):
    image_id: str
    top_k: int = 5


class FilterRequest(BaseModel):
    """Request model for metadata-only filtering"""
    category: Optional[str] = None  # "ring", "necklace"
    metal: Optional[str] = None  # "gold", "silver", "platinum", "white gold", "rose gold"
    primary_stone: Optional[str] = None  # "diamond", "emerald", "ruby", "sapphire", "pearl", etc.
    form: Optional[str] = None  # "plain", "studded"
    top_k: int = 12  # Number of results to return
    page: int = 1
    page_size: int = 12

# %%
# ============================================================
# UTILITY FUNCTIONS FOR DYNAMIC RETRIEVAL
# ============================================================

def adaptive_k_per_query(query: str) -> int:
    """
    Determine optimal retrieval depth based on query characteristics.
    Short/vague queries need more recall, specific queries need precision.
    """
    words = query.lower().split()
    
    # Short/vague queries need more recall
    if len(words) <= 2:
        return 20
    
    # Specific attribute queries
    specific_terms = {'diamond', 'gold', 'silver', 'platinum', 'ring', 'necklace', 'earring', 'bracelet'}
    if any(term in words for term in specific_terms):
        return 15
    
    # Long descriptive queries (high precision)
    if len(words) >= 5:
        return 10
    
    # Default balanced
    return 12


def apply_safety_caps(top_k: int, k_per_query: int, max_candidates: int) -> tuple:
    """
    Apply safety caps to prevent timeouts and abuse.
    HF Spaces has 25s timeout, so we cap aggressively.
    """
    top_k = min(max(top_k, 1), 50)  # 1-50 range
    k_per_query = min(max(k_per_query, 5), 25)  # 5-25 range
    max_candidates = min(max(max_candidates, 10), 100)  # 10-100 range
    
    return top_k, k_per_query, max_candidates


def paginate_results(results: List[Dict], page: int, page_size: int) -> Dict:
    """
    Paginate results and return with metadata.
    """
    total_results = len(results)
    total_pages = max(1, (total_results + page_size - 1) // page_size)
    
    # Clamp page to valid range
    page = max(1, min(page, total_pages))
    
    # Calculate slice
    start = (page - 1) * page_size
    end = start + page_size
    
    return {
        "results": results[start:end],
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_results": total_results,
            "total_pages": total_pages,
            "has_next": end < total_results,
            "has_prev": page > 1
        }
    }


# %%
# ============================================================
# CLIP QUERY ENCODING (TEXT ONLY)
# ============================================================

def encode_text_clip(text: str) -> np.ndarray:
    """Encode text using CLIP with memory cleanup"""
    model = get_clip_model()
    tokens = clip.tokenize([text]).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        result = emb.cpu().numpy()[0]
    
    # Memory cleanup
    del tokens, emb
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    return result


def encode_uploaded_image(image_bytes: bytes) -> np.ndarray:
    """Encode uploaded image using CLIP"""
    from PIL import Image
    import io
    
    model = get_clip_model()
    preprocess = clip.load("ViT-B/16", device=DEVICE)[1]
    
    # Load and preprocess image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        emb = model.encode_image(image_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        result = emb.cpu().numpy()[0]
    
    # Memory cleanup
    del image_tensor, emb
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    return result


# %%
# ============================================================
# MULTI-QUERY RETRIEVAL (QUERY EXPANSION)
# ============================================================

def expand_query_with_llm(query: str, num_variants: int = 5) -> List[str]:
    """
    Use LLM to expand query into multiple semantic variants.
    Helps capture different ways to describe the same jewelry item.
    """
    try:
        prompt = f"""You are a jewelry search expert. Generate {num_variants - 1} alternative search queries for:

Original query: "{query}"

Generate queries that:
1. Use different terminology (e.g., "ring" → "band", "necklace" → "chain")
2. Add relevant style descriptors (e.g., "elegant", "classic", "sophisticated")
3. Vary specificity (more specific and more general)
4. Focus ONLY on product attributes and materials

IMPORTANT:
- Do NOT include occasions or events (no "engagement", "wedding", "bridal")
- Do NOT add contextual tokens that dilute attribute matching
- Focus on physical attributes: metal, stone, form, style

Return ONLY the alternative queries, one per line. No numbering or explanation."""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
            timeout=5.0  # Fail fast for HF Spaces
        )
        
        # Parse numbered list
        content = response.choices[0].message.content.strip()
        expanded = []
        
        for line in content.split('\n'):
            query_text = line.strip().strip('"').strip("'")
            if query_text and len(query_text) > 3:
                expanded.append(query_text)
        
        # Always include original query first
        result = [query] + expanded[:num_variants - 1]  # Max num_variants total
        print(f"🔍 Expanded query into {len(result)} variants: {result}")
        return result
        
    except Exception as e:
        print(f"⚠️ Query expansion failed: {e}, using original query only")
        return [query]  # Fallback to original


def multi_query_retrieval(queries: List[str], k_per_query: int = 12, where_filter: Optional[Dict] = None) -> List[Dict]:
    """
    Retrieve candidates for multiple queries and merge results.
    
    Args:
        queries: List of query strings
        k_per_query: Number of candidates to retrieve per query
        where_filter: Optional ChromaDB where clause for pre-filtering
    
    Returns:
        Merged and deduplicated list of candidates (max 80)
    """
    all_candidates = {}
    
    for idx, query in enumerate(queries, 1):
        try:
            # Encode with CLIP
            embedding = encode_text_clip(query)
            
            # Retrieve from ChromaDB
            results = image_collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=k_per_query,
                where=where_filter  # Apply strict pre-filtering if provided
            )
            
            retrieved_count = len(results['ids'][0])
            print(f"  📌 Query {idx}/{len(queries)}: '{query}' → {retrieved_count} candidates")
            
            # Add to candidates (keep best score per image_id)
            new_candidates = 0
            for img_id, distance in zip(results['ids'][0], results['distances'][0]):
                if img_id not in all_candidates:
                    new_candidates += 1
                    all_candidates[img_id] = {
                        'image_id': img_id,
                        'visual_score': float(distance),
                        'source_query': query
                    }
                elif distance < all_candidates[img_id]['visual_score']:
                    # Update with better score
                    all_candidates[img_id]['visual_score'] = float(distance)
                    all_candidates[img_id]['source_query'] = query
            
            print(f"     ✓ Added {new_candidates} new unique candidates")
        
        except Exception as e:
            print(f"⚠️ Retrieval failed for query '{query}': {e}")
            continue
    
    # Convert to list and sort by similarity
    merged = list(all_candidates.values())
    merged.sort(key=lambda x: x['visual_score'])
    
    # Cap at 80 candidates
    max_candidates = 80
    if len(merged) > max_candidates:
        merged = merged[:max_candidates]
    
    print(f"\n📊 Multi-query retrieval summary:")
    print(f"   • {len(queries)} queries processed")
    print(f"   • {len(all_candidates)} unique candidates found")
    print(f"   • Returning top {len(merged)} candidates\n")
    
    return merged


# %%
# ============================================================
# FIXED: CHROMADB WHERE FILTER BUILDER
# ============================================================

def build_chromadb_where_filter(filters: Dict) -> Optional[Dict]:
    """
    Build ChromaDB-compatible where clause for pre-filtering.
    
    ChromaDB where syntax:
    - Simple: {"category": "ring"}
    - AND: {"$and": [{"category": "ring"}, {"metal": "gold"}]}
    
    Only pre-filter on high-confidence attributes (category, metal).
    Stones are handled in post-filtering for flexibility.
    """
    conditions = []
    
    # Only include EXACT MATCH filters
    if filters.get("category"):
        conditions.append({"category": filters["category"]})
    
    if filters.get("metal"):
        conditions.append({"metal": filters["metal"]})
    
    # Don't pre-filter on stones - they're too noisy
    # We'll handle stones in post-filtering with confidence awareness
    
    if not conditions:
        return None
    
    if len(conditions) == 1:
        return conditions[0]
    
    # ChromaDB AND syntax
    return {"$and": conditions}


# %%
# ============================================================
# FIXED: UNIFIED METADATA FILTERS
# ============================================================

def apply_unified_filters(
    candidates: List[Dict],
    filters: Dict,
    metadata_map: Dict
) -> List[Dict]:
    """
    UNIFIED filtering with confidence-aware soft filtering.
    
    Pipeline:
    1. Hard exclusions (user explicitly said "no X")
    2. Soft inclusion boosts (matches = better rank)
    3. Soft penalties (mismatches with confidence weighting)
    4. Similarity threshold by mode
    5. Minimum results guarantee
    
    Returns:
        Filtered candidates with filter_boost and filter_penalty scores
    """
    
    if not candidates:
        return []
    
    similarity_mode = filters.get("similarity_mode", "balanced")
    filtered = []
    
    excluded_count = 0
    boost_count = 0
    penalty_count = 0
    
    for c in candidates:
        meta = metadata_map.get(c["image_id"])
        if not meta:
            continue
        
        # ============================================
        # PHASE 1: HARD EXCLUSIONS (user said "no X")
        # ============================================
        should_exclude = False
        
        # Exclude by category
        if filters.get("exclude_category"):
            if meta.get("category") == filters["exclude_category"]:
                should_exclude = True
                excluded_count += 1
        
        # Exclude by metal (FIXED: confidence-aware)
        if filters.get("exclude_metal") and not should_exclude:
            meta_metal = meta.get("metal")
            conf_metal = meta.get("confidence_metal", 0)
            
            # Only exclude if we're confident this is the excluded metal
            if meta_metal == filters["exclude_metal"] and conf_metal > 0.7:
                should_exclude = True
                excluded_count += 1
        
        # Exclude by stone
        if filters.get("exclude_stone") and not should_exclude:
            meta_stone = meta.get("primary_stone")
            conf_stone = meta.get("confidence_primary_stone", 0)
            
            if filters["exclude_stone"] == "any":
                # Exclude items WITH stones (plain jewelry)
                if meta_stone and meta_stone not in ["unknown", "null", ""] and conf_stone > 0.6:
                    should_exclude = True
                    excluded_count += 1
            else:
                # Exclude specific stone
                if meta_stone == filters["exclude_stone"] and conf_stone > 0.7:
                    should_exclude = True
                    excluded_count += 1
        
        if should_exclude:
            continue
        
        # ============================================
        # PHASE 2: SOFT INCLUSION BOOSTS & PENALTIES
        # ============================================
        boost = 0.0
        penalty = 0.0
        
        # Category boost/penalty
        if filters.get("category"):
            if meta.get("category") == filters["category"]:
                boost += 0.15
                boost_count += 1
            else:
                conf = meta.get("confidence_category", 0)
                if conf > 0.85:
                    penalty += 0.10
                    penalty_count += 1
                else:
                    penalty += 0.05  # Softer penalty for low confidence
                    penalty_count += 1
        
        # Metal boost/penalty (FIXED: more lenient for uncertain metadata)
        if filters.get("metal"):
            if meta.get("metal") == filters["metal"]:
                boost += 0.20  # Strong boost for exact match
                boost_count += 1
            else:
                conf = meta.get("confidence_metal", 0)
                # Only penalize if we're confident about the metal
                if conf > 0.75:
                    penalty += 0.12
                    penalty_count += 1
                else:
                    penalty += 0.05  # Light penalty for uncertain metadata
                    penalty_count += 1
        
        # Stone boost/penalty (FIXED: handle noisy metadata gracefully)
        if filters.get("primary_stone"):
            if meta.get("primary_stone") == filters["primary_stone"]:
                boost += 0.18
                boost_count += 1
            else:
                conf = meta.get("confidence_primary_stone", 0)
                # Very conservative penalty - stones are hard to detect
                if conf > 0.85:
                    penalty += 0.08
                    penalty_count += 1
                else:
                    penalty += 0.03
                    penalty_count += 1
        
        # Form boost (always soft)
        if filters.get("form"):
            if meta.get("form") == filters["form"]:
                boost += 0.10
                boost_count += 1
            else:
                penalty += 0.05
                penalty_count += 1
        
        # Store adjustments
        c["filter_boost"] = boost
        c["filter_penalty"] = penalty
        
        filtered.append(c)
    
    print(f"📊 Filter stats:")
    print(f"   • Hard exclusions: {excluded_count}")
    print(f"   • Soft boosts applied: {boost_count}")
    print(f"   • Soft penalties applied: {penalty_count}")
    print(f"   • Survived filtering: {len(filtered)}/{len(candidates)}")
    
    # ============================================
    # PHASE 3: SIMILARITY THRESHOLD BY MODE
    # ============================================
    thresholds = {
        "strict": 1.35,
        "balanced": 1.9,
        "exploratory": 3.2
    }
    threshold = thresholds.get(similarity_mode, 1.9)
    
    sim_filtered = [c for c in filtered if c["visual_score"] < threshold]
    
    # Fallback if strict mode returns nothing
    if similarity_mode == "strict" and len(sim_filtered) < 3:
        print("⚠️ Strict mode too restrictive → falling back to balanced")
        threshold = thresholds["balanced"]
        sim_filtered = [c for c in filtered if c["visual_score"] < threshold]
    
    # ============================================
    # PHASE 4: MINIMUM RESULTS GUARANTEE
    # ============================================
    MIN_RESULTS = 6
    if len(sim_filtered) < MIN_RESULTS and len(filtered) >= MIN_RESULTS:
        print(f"⚠️ Only {len(sim_filtered)} results → ensuring minimum {MIN_RESULTS}")
        # Sort by visual score and take top N
        sim_filtered = sorted(filtered, key=lambda x: x["visual_score"])[:MIN_RESULTS]
    
    print(f"   • After similarity filter ({similarity_mode}): {len(sim_filtered)}\n")
    
    return sim_filtered


# %%
# ============================================================
# FIXED: SINGLE SCORE FUSION POINT
# ============================================================

def compute_final_scores(
    candidates: List[Dict],
    use_reranking: bool = True
) -> List[Dict]:
    """
    Compute final scores with clear fusion formula.
    
    Score components (lower = better):
    - Visual similarity (CLIP): 40%
    - Filter alignment (metadata): 20%
    - Semantic relevance (cross-encoder): 40%
    
    Formula:
    final_score = 0.40 * visual + 0.20 * (visual + penalty - boost) + 0.40 * cross_sim
    
    Returns:
        Candidates sorted by final_score (ascending)
    """
    
    for c in candidates:
        visual = c.get("visual_score", 0)
        boost = c.get("filter_boost", 0)
        penalty = c.get("filter_penalty", 0)
        cross_score = c.get("cross_encoder_score", 0)
        
        if use_reranking and cross_score != 0:
            # With cross-encoder (inverted scale: higher cross_score = better)
            # Convert cross_score to similarity scale (lower = better)
            cross_similarity = 1.0 - cross_score
            
            c["final_score"] = (
                0.40 * visual +                          # Visual similarity
                0.20 * (visual + penalty - boost) +      # Filter-adjusted visual
                0.40 * cross_similarity                  # Semantic similarity
            )
        else:
            # Without cross-encoder (visual + filter adjustments only)
            c["final_score"] = visual + penalty - boost
    
    return sorted(candidates, key=lambda x: x["final_score"])


# %%
# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def rerank_with_cross_encoder(
    query: str,
    candidates: List[Dict],
    top_k: int = 12
) -> List[Dict]:
    """
    Re-rank candidates using cross-encoder for better semantic matching.
    
    Args:
        query: User's search query
        candidates: List of candidate results
        top_k: Number of top results to return
    
    Returns:
        Re-ranked candidates with cross_encoder_score
    """

    if not candidates:
        return []

    pairs = []

    for c in candidates:
        # Fetch metadata
        meta = metadata_collection.get(
            ids=[c["image_id"]],
            include=["metadatas"]
        )["metadatas"][0]

        enhanced_meta = ENHANCED_METADATA.get(c["image_id"], {})
        
        # Use Florence caption if available, otherwise fall back to metadata_text
        florence_caption = FLORENCE_CAPTIONS.get(c["image_id"], "")
        
        if florence_caption:
            # Use Florence caption for richer semantic matching
            doc_text = florence_caption
        else:
            # Fallback to metadata_text
            doc_text = enhanced_meta.get(
                "metadata_text",
                f"{meta.get('category','jewellery')} made of {meta.get('metal','metal')}"
            )

        pairs.append([query, doc_text])

    # Cross-encoder scoring
    print(f"🔄 Cross-encoder scoring {len(pairs)} candidates...")
    encoder = get_cross_encoder()

    cross_scores = encoder.predict(
        pairs,
        batch_size=32
    )

    # Add cross-encoder scores to candidates
    for i, c in enumerate(candidates):
        c["cross_encoder_score"] = float(cross_scores[i])

    print(f"✅ Cross-encoder scored {len(candidates)} candidates")

    return candidates  # Don't sort here, let compute_final_scores handle it


# %%
# ============================================================
# INTENT & ATTRIBUTE DETECTION WITH LLM
# ============================================================

def detect_intent_and_attributes(query: str) -> Dict:
    """
    Extract search attributes and exclusions from query using LLM with fixed schema.
    
    Returns:
        {
            "intent": "search",
            "attributes": {category, metal, primary_stone},  # Items to INCLUDE
            "exclusions": {category, metal, primary_stone}   # Items to EXCLUDE
        }
    """
    
    prompt = f"""Extract jewellery search attributes from this query.

Query: "{query}"

Return ONLY valid JSON with this exact schema:
{{
  "intent": "search",
  "attributes": {{
    "category": "ring|necklace|earring|bracelet|null",
    "metal": "gold|silver|platinum|null",
    "primary_stone": "diamond|pearl|ruby|emerald|sapphire|null"
  }},
  "exclusions": {{
    "category": "ring|necklace|earring|bracelet|null",
    "metal": "gold|silver|platinum|null",
    "primary_stone": "diamond|pearl|ruby|emerald|sapphire|null"
  }}
}}

Rules:
- "attributes" = what to INCLUDE (positive filters)
- "exclusions" = what to EXCLUDE (negative filters)
- Use null for unspecified fields
- Detect negations: "no", "without", "not", "plain", "-free"

Examples:

Query: "gold ring with diamonds"
{{"intent": "search", "attributes": {{"category": "ring", "metal": "gold", "primary_stone": "diamond"}}, "exclusions": {{"category": null, "metal": null, "primary_stone": null}}}}

Query: "ring with no diamonds"
{{"intent": "search", "attributes": {{"category": "ring", "metal": null, "primary_stone": null}}, "exclusions": {{"category": null, "metal": null, "primary_stone": "diamond"}}}}

Query: "plain silver necklace"
{{"intent": "search", "attributes": {{"category": "necklace", "metal": "silver", "primary_stone": null}}, "exclusions": {{"category": null, "metal": null, "primary_stone": "any"}}}}

Return ONLY the JSON, no explanation."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        # Use regex to extract only the first JSON object (handles extra text after JSON)
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(0)
        
        result = json.loads(result_text)
        
        # Clean null values
        result["attributes"] = {k: v for k, v in result.get("attributes", {}).items() if v and v != "null"}
        result["exclusions"] = {k: v for k, v in result.get("exclusions", {}).items() if v and v != "null"}
        
        return result
        
    except Exception as e:
        print(f"⚠️ LLM extraction failed: {e}, falling back to simple extraction")
        # Fallback to simple keyword matching
        q = query.lower()
        attrs = {}
        
        if "necklace" in q:
            attrs["category"] = "necklace"
        elif "ring" in q:
            attrs["category"] = "ring"
        
        if "gold" in q:
            attrs["metal"] = "gold"
        elif "silver" in q:
            attrs["metal"] = "silver"
        
        if "pearl" in q:
            attrs["primary_stone"] = "pearl"
        elif "diamond" in q:
            attrs["primary_stone"] = "diamond"
        
        return {
            "intent": "search",
            "attributes": attrs,
            "exclusions": {}
        }


# %%
# ============================================================
# LLM-POWERED EXPLANATION (BATCH PROCESSING)
# ============================================================

def batch_generate_explanations(results: List[Dict], query_attrs: Dict, user_query: str) -> List[str]:
    """Generate detailed, LLM-powered explanations for all search results in ONE API call"""
    
    if not results:
        return []
    
    # Build rich context for all items
    items_context = []
    for idx, r in enumerate(results, 1):
        meta = metadata_collection.get(
            ids=[r["image_id"]],
            include=["metadatas"]
        )["metadatas"][0]
        
        # Get enhanced metadata if available
        enhanced_meta = ENHANCED_METADATA.get(r["image_id"], {})
        
        matched_attrs = [v for k, v in query_attrs.items() if meta.get(k) == v]
        
        category = meta.get('category', 'item').title()
        metal = meta.get('metal', 'unknown')
        stone = meta.get('primary_stone', 'no stones')
        visual_score = r.get('visual_score', 0)
        cross_score = r.get('cross_encoder_score', 0)
        
        # Rich format with more context
        item_info = f"""{idx}. {category}
   Material: {metal} with {stone}
   Match: visual={visual_score:.2f}, semantic={cross_score:.2f}
   Attributes: {', '.join(matched_attrs) if matched_attrs else 'classic design'}
   Features: {enhanced_meta.get('metadata_text', 'premium craftsmanship')[:80]}"""
        items_context.append(item_info)
    
    # Enhanced prompt for detailed, engaging descriptions
    prompt = f"""You are a luxury jewelry expert at Tanishq. For each item below, write a detailed, engaging 2-3 sentence description.

Query: "{user_query}"

For each item, highlight:
- Why it matches the query
- Unique design features and craftsmanship
- Material quality and aesthetic appeal
- Who would love this piece

Items:
{chr(10).join(items_context)}

Format:
1. [2-3 detailed, engaging sentences]
2. [2-3 detailed, engaging sentences]
etc."""

    try:
        # Try with current API key, with fallback to other keys on rate limit
        last_error = None
        
        for attempt, key_index in enumerate(range(len(GROQ_API_KEYS))):
            try:
                # Rotate to next key if not first attempt
                if attempt > 0:
                    global current_groq_key_index
                    current_groq_key_index = (current_groq_key_index + 1) % len(GROQ_API_KEYS)
                    groq_client.api_key = GROQ_API_KEYS[current_groq_key_index]
                    print(f"🔄 Retrying with fallback API key #{current_groq_key_index + 1}")
                
                # Single API call for ALL items
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are an expert jewelry consultant at Tanishq, India's most trusted jewelry brand. Write detailed, engaging descriptions that highlight craftsmanship, design, and emotional appeal."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=min(1500, len(results) * 120),
                    top_p=0.9,
                    timeout=10.0  # 10 second timeout
                )
                
                full_response = response.choices[0].message.content.strip()
                
                explanations = []
                
                import re
                # Try multiple patterns to be more flexible
                pattern = r'^\s*(\d+)[\.:)\-]\s*(.+?)(?=^\s*\d+[\.:)\-]|\Z)'
                matches = re.findall(pattern, full_response, re.MULTILINE | re.DOTALL)
                
                if matches and len(matches) >= len(results):
                    for num, text in matches[:len(results)]:
                        clean_text = ' '.join(text.strip().split())
                        if clean_text and len(clean_text) > 10:
                            explanations.append(clean_text)
                else:
                    # Try simpler line-by-line parsing
                    lines = full_response.split('\n')
                    current_explanation = []
                    
                    for line in lines:
                        line = line.strip()
                        # Remove markdown bold markers
                        line_clean = line.replace('**', '').strip()
                        
                        # Check if line starts with a number (after removing markdown)
                        if re.match(r'^\d+[\.:)\-]', line_clean):
                            # Save previous explanation if exists
                            if current_explanation:
                                exp_text = ' '.join(current_explanation)
                                if len(exp_text) > 10:
                                    explanations.append(exp_text)
                            # Start new explanation (remove number prefix and markdown)
                            current_explanation = [re.sub(r'^\d+[\.:)\-]\s*', '', line_clean)]
                        elif line_clean and current_explanation:
                            # Continue current explanation
                            current_explanation.append(line_clean)
                    
                    # Add last explanation
                    if current_explanation:
                        exp_text = ' '.join(current_explanation)
                        if len(exp_text) > 10:
                            explanations.append(exp_text)
                
                print(f"✅ Parsed {len(explanations)} explanations from LLM response")
                
                if len(explanations) >= len(results):
                    return explanations[:len(results)]
                
                # If incomplete, pad with fallback
                print(f"⚠️ LLM returned {len(explanations)}/{len(results)} explanations, padding with fallback")
                break  # Success, exit retry loop
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                if "rate" in error_str or "429" in error_str:
                    print(f"⚠️ Rate limit hit on API key #{current_groq_key_index + 1}: {e}")
                    if attempt < len(GROQ_API_KEYS) - 1:
                        continue  # Try next key
                    else:
                        print(f"❌ All {len(GROQ_API_KEYS)} API keys exhausted due to rate limits")
                        break
                
                # Check if it's a timeout
                elif "timeout" in error_str or "timed out" in error_str:
                    print(f"⏱️ LLM request timed out after 10 seconds: {e}")
                    break  # Don't retry on timeout
                
                else:
                    print(f"⚠️ LLM explanation failed: {e}")
                    break  # Don't retry on other errors
        
        # If we got here, use fallback
        if last_error:
            print(f"⚠️ Using fallback explanations after error: {last_error}")
        explanations = []
        
    except Exception as e:
        print(f"⚠️ Unexpected error in LLM explanation: {e}, using fallback")
        explanations = []
    
    # Fallback for missing explanations
    while len(explanations) < len(results):
        idx = len(explanations)
        r = results[idx]
        meta = metadata_collection.get(
            ids=[r["image_id"]],
            include=["metadatas"]
        )["metadatas"][0]
        matched_attrs = [v for k, v in query_attrs.items() if meta.get(k) == v]
        
        category = meta.get('category', 'item')
        metal = meta.get('metal', 'unknown')
        stone = meta.get('primary_stone', 'unknown')
        
        # Enhanced fallback explanations with more detail
        cross_score = r.get('cross_encoder_score', 0)
        
        if matched_attrs and cross_score > 0.5:
            explanations.append(
                f"This exquisite {metal} {category} perfectly matches your search for {' and '.join(matched_attrs)}. "
                f"The {stone} stones are expertly set, creating a timeless piece that combines elegance with craftsmanship. "
                f"Ideal for those seeking authentic Tanishq quality with exceptional visual similarity."
            )
        elif matched_attrs and r['visual_score'] < 1.3:
            explanations.append(
                f"A stunning {category} crafted in {metal}, featuring beautiful {stone}. "
                f"This piece showcases {' and '.join(matched_attrs)}, making it a perfect match for your preferences. "
                f"The intricate design and premium materials reflect Tanishq's commitment to excellence."
            )
        elif r['visual_score'] < 1.3:
            explanations.append(
                f"This elegant {metal} {category} with {stone} demonstrates exceptional craftsmanship and visual appeal. "
                f"The design captures timeless beauty while maintaining modern sophistication. "
                f"A versatile piece suitable for both everyday wear and special occasions."
            )
        else:
            explanations.append(
                f"A beautifully crafted {metal} {category} adorned with {stone}, showcasing Tanishq's signature attention to detail. "
                f"This piece combines traditional artistry with contemporary design elements. "
                f"Perfect for those who appreciate fine jewelry with lasting value."
            )
    
    return explanations


# %%
# ============================================================
# OCR TEXT EXTRACTION (NVIDIA + GPT-4 FALLBACK)
# ============================================================

async def extract_text_from_image_async(image_bytes: bytes) -> str:
    """
    Extract text from image using OCR.
    Primary: NVIDIA OCR API
    Fallback: OpenAI GPT-4o-mini Vision
    """
    # Try NVIDIA OCR first
    if NVIDIA_API_KEY:
        try:
            print("🔹 Attempting NVIDIA OCR...")
            
            # Convert to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Check image size (NVIDIA has 180k character limit)
            if len(image_b64) >= 180000:
                print("⚠️ Image too large for NVIDIA OCR (>180k chars)")
                raise Exception("Image exceeds NVIDIA OCR size limit")
            
            # NVIDIA OCR API call
            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Accept": "application/json"
            }
            
            payload = {
                "input": [
                    {
                        "type": "image_url",
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                ]
            }
            
            response = requests.post(
                NVIDIA_OCR_URL,
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                # NVIDIA OCR returns data in different format
                extracted_text = result.get("data", [{}])[0].get("content", "")
                
                if extracted_text and len(extracted_text.strip()) > 0:
                    print(f"✅ NVIDIA OCR extracted: '{extracted_text}'")
                    return extracted_text.strip()
            
            print(f"⚠️ NVIDIA OCR failed with status {response.status_code}")
            
        except Exception as e:
            print(f"⚠️ NVIDIA OCR error: {e}")
    
    # Fallback to OpenAI GPT-4o-mini Vision
    if OPENAI_API_KEY:
        try:
            print("🔹 Falling back to OpenAI gpt-4.1-nano Vision OCR...")
            
            from openai import OpenAI
            openai_client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url="https://apidev.navigatelabsai.com/"
            )
            
            # Convert to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            response = openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this image. Return only the extracted text, no explanations or formatting."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0
            )
            
            extracted_text = response.choices[0].message.content.strip()
            
            if extracted_text:
                print(f"✅ OpenAI OCR extracted: '{extracted_text}'")
                return extracted_text
            
        except Exception as e:
            print(f"⚠️ OpenAI OCR error: {e}")
    
    # If both fail
    raise HTTPException(
        status_code=500,
        detail="OCR extraction failed. Both NVIDIA and OpenAI OCR unavailable."
    )


def extract_text_from_image(image_bytes: bytes) -> str:
    """Synchronous wrapper for async OCR extraction"""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If we're already in an async context, create a new loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, extract_text_from_image_async(image_bytes))
            return future.result()
    else:
        return loop.run_until_complete(extract_text_from_image_async(image_bytes))


# %%
# ============================================================
# FIXED: SEARCH TEXT ENDPOINT
# ============================================================

@app.post("/search/text")
def search_text(req: TextSearchRequest):
    """
    Enhanced text search with proper filter application.
    
    Pipeline:
    1. Build ChromaDB pre-filter (category + metal only)
    2. Query expansion (LLM)
    3. Multi-query retrieval WITH pre-filter
    4. Fetch metadata (single batch)
    5. Apply unified filters (exclusions + boosts)
    6. Cross-encoder rerank (optional)
    7. Compute final scores
    8. Paginate
    9. Generate explanations (optional)
    """
    import time
    start_time = time.time()
    
    # Apply safety caps
    top_k, k_per_query, max_candidates = apply_safety_caps(
        req.top_k, req.k_per_query, req.max_candidates
    )
    
    # Adaptive k_per_query
    if req.k_per_query == 12:
        k_per_query = adaptive_k_per_query(req.query)
        print(f"🎯 Adaptive k_per_query: {k_per_query}")
    
    # ============================================
    # STAGE 1: Build ChromaDB pre-filter (FIXED)
    # ============================================
    filters = {
        'category': req.category,
        'metal': req.metal,
        'primary_stone': req.primary_stone,
        'form': req.form,
        'exclude_category': req.exclude_category,
        'exclude_metal': req.exclude_metal,
        'exclude_stone': req.exclude_stone,
        'similarity_mode': req.similarity_mode
    }
    
    where_filter = build_chromadb_where_filter(filters)
    if where_filter:
        print(f"🔍 ChromaDB pre-filter: {where_filter}")
    
    # ============================================
    # STAGE 2: Query expansion + retrieval
    # ============================================
    queries = expand_query_with_llm(req.query)
    
    # Semantic enhancement for gemstones
    if req.primary_stone:
        STONE_DESCRIPTORS = {
            "emerald": "green gemstone",
            "ruby": "red gemstone",
            "sapphire": "blue gemstone",
            "diamond": "clear brilliant",
            "pearl": "white lustrous"
        }
        if req.primary_stone.lower() in STONE_DESCRIPTORS:
            descriptor = STONE_DESCRIPTORS[req.primary_stone.lower()]
            queries = [f"{q} {descriptor}" for q in queries]
            print(f"🎨 Semantic enhancement: Added '{descriptor}' for {req.primary_stone}")
    
    # Multi-query retrieval
    retrieval_start = time.time()
    candidates = multi_query_retrieval(
        queries,
        k_per_query=k_per_query,
        where_filter=where_filter
    )
    
    candidates = candidates[:max_candidates]
    retrieval_time = time.time() - retrieval_start
    
    if not candidates:
        return {
            "results": [],
            "pagination": {
                "page": 1,
                "page_size": req.page_size,
                "total_results": 0,
                "total_pages": 0,
                "has_next": False,
                "has_prev": False
            },
            "message": "No results found. Try relaxing filters or using different keywords.",
            "performance": {
                "total_time_s": round(time.time() - start_time, 2),
                "candidates_retrieved": 0
            }
        }
    
    # ============================================
    # STAGE 3: Fetch ALL metadata in ONE batch (FIXED)
    # ============================================
    candidate_ids = [c["image_id"] for c in candidates]
    metadata_batch = metadata_collection.get(
        ids=candidate_ids,
        include=["metadatas"]
    )
    
    metadata_map = {
        img_id: meta
        for img_id, meta in zip(metadata_batch["ids"], metadata_batch["metadatas"])
    }
    
    # ============================================
    # STAGE 4: Apply unified filters (FIXED)
    # ============================================
    filter_start = time.time()
    filtered = apply_unified_filters(candidates, filters, metadata_map)
    filter_time = time.time() - filter_start
    
    if not filtered:
        return {
            "results": [],
            "pagination": {
                "page": 1,
                "page_size": req.page_size,
                "total_results": 0,
                "total_pages": 0,
                "has_next": False,
                "has_prev": False
            },
            "message": "Filters too restrictive. Try 'balanced' or 'exploratory' mode.",
            "performance": {
                "total_time_s": round(time.time() - start_time, 2),
                "candidates_retrieved": len(candidates),
                "after_filters": 0
            }
        }
    
    # ============================================
    # STAGE 5: Cross-encoder rerank (optional)
    # ============================================
    if req.use_reranking:
        rerank_start = time.time()
        filtered = rerank_with_cross_encoder(
            req.query,
            filtered,
            top_k=min(len(filtered), 80)
        )
        rerank_time = time.time() - rerank_start
    else:
        rerank_time = 0
    
    # ============================================
    # STAGE 6: Compute final scores (FIXED)
    # ============================================
    ranked = compute_final_scores(filtered, req.use_reranking)
    
    # Keep 4x for pagination
    ranked = ranked[:top_k * 4]
    
    # ============================================
    # STAGE 7: Pagination
    # ============================================
    paginated = paginate_results(ranked, req.page, req.page_size)
    page_results = paginated["results"]
    
    # ============================================
    # STAGE 8: Generate explanations (optional)
    # ============================================
    if req.use_explanations:
        intent = detect_intent_and_attributes(req.query)
        attrs = intent["attributes"]
        explanations = batch_generate_explanations(page_results, attrs, req.query)
    else:
        explanations = ["Match found"] * len(page_results)
    
    # ============================================
    # STAGE 9: Format results
    # ============================================
    results = []
    for r, explanation in zip(page_results, explanations):
        results.append({
            "image_id": r["image_id"],
            "explanation": explanation,
            "metadata": ENHANCED_METADATA.get(r["image_id"], {}),
            "scores": {
                "visual": round(r["visual_score"], 3),
                "filter_boost": round(r.get("filter_boost", 0), 3),
                "filter_penalty": round(r.get("filter_penalty", 0), 3),
                "cross_encoder": round(r.get("cross_encoder_score", 0), 3),
                "final": round(r["final_score"], 3)
            }
        })
    
    total_time = time.time() - start_time
    
    return {
        "results": results,
        "pagination": paginated["pagination"],
        "performance": {
            "total_time_s": round(total_time, 2),
            "retrieval_time_s": round(retrieval_time, 2),
            "filter_time_s": round(filter_time, 2),
            "rerank_time_s": round(rerank_time, 2),
            "candidates_retrieved": len(candidates),
            "after_filters": len(filtered),
            "final_ranked": len(ranked)
        },
        "filters_applied": {k: v for k, v in filters.items() if v is not None}
    }


# %%
# ============================================================
# SIMILAR SEARCH ENDPOINT
# ============================================================

@app.post("/search/similar")
def search_similar(req: SimilarSearchRequest):
    """Find similar items based on an existing image"""
    
    base = image_collection.get(
        ids=[req.image_id],
        include=["embeddings"]
    )["embeddings"][0]

    res = image_collection.query(
        query_embeddings=[base],
        n_results=req.top_k + 1
    )

    base_meta = metadata_collection.get(
        ids=[req.image_id],
        include=["metadatas"]
    )["metadatas"][0]

    attrs = {
        k: base_meta[k]
        for k in ["category", "metal", "primary_stone"]
        if base_meta.get(k) != "unknown"
    }

    candidates = [
        {
            "image_id": img_id,
            "visual_score": dist,
            "filter_boost": 0,
            "filter_penalty": 0
        }
        for img_id, dist in zip(res["ids"][0], res["distances"][0])
        if img_id != req.image_id
    ]

    # Compute final scores
    ranked = compute_final_scores(candidates, use_reranking=False)[:req.top_k]

    # Generate explanations
    query_context = f"items similar to {req.image_id}"
    explanations = batch_generate_explanations(ranked, attrs, query_context)
    
    results = []
    for r, explanation in zip(ranked, explanations):
        results.append({
            "image_id": r["image_id"],
            "explanation": explanation,
            "scores": {
                "visual": round(r["visual_score"], 3),
                "final": round(r["final_score"], 3)
            }
        })

    return {
        "base_image": req.image_id,
        "results": results
    }


# %%
# ============================================================
# IMAGE UPLOAD SEARCH ENDPOINT
# ============================================================

@app.post("/search/upload-image")
async def search_by_uploaded_image(
    file: UploadFile = File(...),
    top_k: int = 12
):
    """
    Search for similar jewellery items by uploading an image.
    The image is encoded using CLIP and queried against the database.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Encode image with CLIP
        query_embedding = encode_uploaded_image(image_bytes)
        
        # Query ChromaDB
        res = image_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(100, top_k * 10),
            include=["distances"]
        )
        
        # Get metadata for all results
        candidates = []
        for img_id, dist in zip(res["ids"][0], res["distances"][0]):
            candidates.append({
                "image_id": img_id,
                "visual_score": dist,
                "filter_boost": 0,
                "filter_penalty": 0
            })
        
        # Get metadata from first result to infer attributes
        if candidates:
            base_meta = metadata_collection.get(
                ids=[candidates[0]["image_id"]],
                include=["metadatas"]
            )["metadatas"][0]
            
            attrs = {
                k: base_meta[k]
                for k in ["category", "metal", "primary_stone"]
                if base_meta.get(k) != "unknown"
            }
        else:
            attrs = {}
        
        # Compute final scores
        ranked = compute_final_scores(candidates, use_reranking=False)[:top_k]
        
        # Generate explanations
        query_context = f"items visually similar to uploaded image"
        explanations = batch_generate_explanations(ranked, attrs, query_context)
        
        results = []
        for r, explanation in zip(ranked, explanations):
            results.append({
                "image_id": r["image_id"],
                "explanation": explanation,
                "scores": {
                    "visual": round(r["visual_score"], 3),
                    "final": round(r["final_score"], 3)
                }
            })
        
        return {
            "query_type": "uploaded_image",
            "filename": file.filename,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")


# %%
# ============================================================
# OCR QUERY SEARCH ENDPOINT
# ============================================================

@app.post("/search/ocr-query")
async def search_by_ocr_query(
    file: UploadFile = File(...),
    top_k: int = 12,
    use_reranking: bool = True,
    use_explanations: bool = True,
    # Filters (optional)
    category: Optional[str] = None,
    metal: Optional[str] = None,
    primary_stone: Optional[str] = None,
    form: Optional[str] = None,
    exclude_category: Optional[str] = None,
    exclude_metal: Optional[str] = None,
    exclude_stone: Optional[str] = None,
    confidence_category: Optional[float] = None,
    confidence_metal: Optional[float] = None,
    confidence_primary_stone: Optional[float] = None,
    similarity_mode: str = "balanced"
):
    """
    Extract text from uploaded image using OCR,
    then perform enhanced text search with multi-query retrieval and filters.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Extract text using OCR (NVIDIA → OpenAI fallback)
        extracted_text = extract_text_from_image(image_bytes)
        
        print(f"📝 Extracted text from image: '{extracted_text}'")
        
        # Build filters
        filters = {
            'category': category,
            'metal': metal,
            'primary_stone': primary_stone,
            'form': form,
            'exclude_category': exclude_category,
            'exclude_metal': exclude_metal,
            'exclude_stone': exclude_stone,
            'similarity_mode': similarity_mode
        }
        
        where_filter = build_chromadb_where_filter(filters)
        
        # Stage 1: Query expansion
        queries = expand_query_with_llm(extracted_text)
        
        # Stage 2: Multi-query retrieval
        candidates = multi_query_retrieval(queries, k_per_query=12, where_filter=where_filter)
        candidates = candidates[:80]
        
        if not candidates:
            return {
                "extracted_text": extracted_text,
                "results": [],
                "message": "No results found for OCR query"
            }
        
        # Stage 3: Fetch metadata
        candidate_ids = [c["image_id"] for c in candidates]
        metadata_batch = metadata_collection.get(
            ids=candidate_ids,
            include=["metadatas"]
        )
        metadata_map = {
            img_id: meta
            for img_id, meta in zip(metadata_batch["ids"], metadata_batch["metadatas"])
        }
        
        # Stage 4: Apply filters
        filtered = apply_unified_filters(candidates, filters, metadata_map)
        
        if not filtered:
            return {
                "extracted_text": extracted_text,
                "results": [],
                "message": "Filters too restrictive"
            }
        
        # Stage 5: Cross-encoder re-ranking
        if use_reranking:
            filtered = rerank_with_cross_encoder(extracted_text, filtered, min(len(filtered), 80))
        
        # Stage 6: Compute final scores
        ranked = compute_final_scores(filtered, use_reranking)[:top_k]
        
        # Stage 7: Generate explanations
        if use_explanations:
            intent = detect_intent_and_attributes(extracted_text)
            attrs = intent["attributes"]
            explanations = batch_generate_explanations(ranked, attrs, extracted_text)
        else:
            explanations = ["Match found"] * len(ranked)
        
        results = []
        for r, explanation in zip(ranked, explanations):
            results.append({
                "image_id": r["image_id"],
                "explanation": explanation,
                "scores": {
                    "visual": round(r["visual_score"], 3),
                    "filter_boost": round(r.get("filter_boost", 0), 3),
                    "filter_penalty": round(r.get("filter_penalty", 0), 3),
                    "cross_encoder": round(r.get("cross_encoder_score", 0), 3),
                    "final": round(r["final_score"], 3)
                }
            })
        
        return {
            "extracted_text": extracted_text,
            "expanded_queries": queries,
            "filters_applied": {k: v for k, v in filters.items() if v is not None},
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ OCR search error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR search failed: {str(e)}")


# %%
# ============================================================
# METADATA-ONLY FILTERING ENDPOINT
# ============================================================

@app.post("/filter/metadata")
def filter_by_metadata(req: FilterRequest):
    """
    Fast metadata-only filtering without embeddings or reranking.
    
    Directly queries ChromaDB metadata for exact matches.
    Perfect for simple category/metal/stone filtering.
    
    Returns results sorted by confidence scores.
    """
    import time
    start_time = time.time()
    
    try:
        # Build metadata filter
        filter_conditions = []
        filters_applied = {}
        
        if req.category:
            filter_conditions.append({"category": req.category})
            filters_applied["category"] = req.category
            
        if req.metal:
            filter_conditions.append({"metal": req.metal})
            filters_applied["metal"] = req.metal
            
        if req.primary_stone:
            filter_conditions.append({"primary_stone": req.primary_stone})
            filters_applied["primary_stone"] = req.primary_stone
            
        if req.form:
            filter_conditions.append({"form": req.form.lower()})
            filters_applied["form"] = req.form
        
        # If no filters provided, return error
        if not filter_conditions:
            raise HTTPException(
                status_code=400,
                detail="At least one filter (category, metal, primary_stone, or form) must be provided"
            )
        
        # Build where clause
        if len(filter_conditions) == 1:
            where_filter = filter_conditions[0]
        else:
            where_filter = {"$and": filter_conditions}
        
        print(f"🎛️ Metadata-only filter: {where_filter}")
        
        # Determine query limit based on filter type
        specific_stones = ['emerald', 'ruby', 'sapphire', 'amethyst', 'topaz', 
                          'aquamarine', 'onyx', 'citrine', 'pearl']
        
        if req.primary_stone and req.primary_stone.lower() in specific_stones:
            query_limit = 1000  # Return ALL items for specific stones
            print(f"🔍 Specific stone filter ({req.primary_stone}) - returning ALL matching items")
        else:
            query_limit = req.top_k * 3
        
        # Query ChromaDB metadata collection
        results = metadata_collection.get(
            where=where_filter,
            limit=query_limit
        )
        
        # Extract results
        image_ids = results["ids"]
        metadatas = results["metadatas"]
        
        print(f"📊 Found {len(image_ids)} results")
        
        # Combine results with metadata
        combined_results = []
        for img_id, meta in zip(image_ids, metadatas):
            # Calculate overall confidence
            confidences = [
                meta.get("confidence_category", 0.5),
                meta.get("confidence_metal", 0.5),
                meta.get("confidence_primary_stone", 0.5),
                meta.get("confidence_form", 0.5)
            ]
            overall_confidence = sum(confidences) / len(confidences)
            
            combined_results.append({
                "image_id": img_id,
                "metadata": meta,
                "confidence": overall_confidence
            })
        
        # Sort by confidence (highest first)
        combined_results.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Apply pagination
        start_idx = (req.page - 1) * req.page_size
        end_idx = start_idx + req.page_size
        paginated_results = combined_results[start_idx:end_idx]
        
        # Format response
        formatted_results = []
        for r in paginated_results:
            formatted_results.append({
                "image_id": r["image_id"],
                "metadata": {
                    "category": r["metadata"].get("category"),
                    "metal": r["metadata"].get("metal"),
                    "primary_stone": r["metadata"].get("primary_stone"),
                    "form": r["metadata"].get("form"),
                    "confidence_category": r["metadata"].get("confidence_category"),
                    "confidence_metal": r["metadata"].get("confidence_metal"),
                    "confidence_primary_stone": r["metadata"].get("confidence_primary_stone"),
                    "confidence_overall": r["confidence"]
                }
            })
        
        elapsed = time.time() - start_time
        print(f"⏱️  Metadata filtering: {elapsed:.3f}s")
        
        return {
            "results": formatted_results,
            "total": len(combined_results),
            "page": req.page,
            "page_size": req.page_size,
            "filters_applied": filters_applied,
            "elapsed_ms": int(elapsed * 1000)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Metadata filtering error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Metadata filtering failed: {str(e)}")


# %%
# ============================================================
# HEALTH CHECK & IMAGE SERVING
# ============================================================

@app.get("/health")
def health_check():
    """Health check endpoint for HF Spaces monitoring"""
    return {
        "status": "healthy",
        "models_loaded": {
            "clip": clip_model is not None,
            "cross_encoder": cross_encoder is not None,
            "enhanced_metadata": len(ENHANCED_METADATA) > 0
        },
        "database": {
            "images": image_collection.count(),
            "metadata": metadata_collection.count()
        }
    }


@app.get("/image/{image_id}")
def get_image(image_id: str):
    """Serve image file"""
    path = os.path.join(IMAGE_DIR, image_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


# %%
# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Jewellery Search API server...")
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"📁 Image directory: {IMAGE_DIR}")
    print(f"📁 ChromaDB path: {CHROMA_PATH}")
    print(f"🌐 Server will run on: http://localhost:8000")
    print(f"📖 API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")