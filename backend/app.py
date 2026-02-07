# %%
# ============================================================
# JEWELLERY MULTIMODAL SEARCH BACKEND (FASTAPI)
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
    # Running locally - use script directory as base
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_primary")
DATA_DIR = os.path.join(BASE_DIR, "data", "tanishq")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
BLIP_CAPTIONS_PATH = os.path.join(DATA_DIR, "blip_captions.json")

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
print("🔹 Loading BLIP captions...")
with open(BLIP_CAPTIONS_PATH, "r") as f:
    BLIP_CAPTIONS = json.load(f)

# %%
# ============================================================
# INITIALIZE GROQ LLM CLIENT
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if GROQ_API_KEY:
    print("🔹 Initializing Groq LLM client...")
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )
else:
    groq_client = None
    print("⚠️ GROQ_API_KEY not set; LLM features disabled (fallbacks enabled)")

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
    
    # Add request timeout (60s for local dev/slower machines)
    try:
        response = await asyncio.wait_for(call_next(request), timeout=60.0)
        return response
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout (max 60s)")

# %%
# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class TextSearchRequest(BaseModel):
    query: str
    filters: Dict[str, str] = None  # Explicit UI filters (e.g. {"metal": "gold"})
    top_k: int = 5
    use_reranking: bool = True  # Toggle cross-encoder (3x faster when False)
    use_explanations: bool = True  # Toggle LLM explanations (500ms+ faster when False)


class SimilarSearchRequest(BaseModel):
    image_id: str
    top_k: int = 5

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

# %%
# ============================================================
# INTENT & ATTRIBUTE DETECTION WITH LLM (STRUCTURED)
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

Query: "gold necklace without pearls"
{{"intent": "search", "attributes": {{"category": "necklace", "metal": "gold", "primary_stone": null}}, "exclusions": {{"category": null, "metal": null, "primary_stone": "pearl"}}}}

Return ONLY the JSON, no explanation."""

    def simple_fallback() -> Dict:
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

    if groq_client is None:
        return simple_fallback()

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
        
        result = json.loads(result_text)
        
        # Clean null values
        result["attributes"] = {k: v for k, v in result.get("attributes", {}).items() if v and v != "null"}
        result["exclusions"] = {k: v for k, v in result.get("exclusions", {}).items() if v and v != "null"}
        
        return result
        
    except Exception as e:
        print(f"⚠️ LLM extraction failed: {e}, falling back to simple extraction")
        return simple_fallback()


# %%
# ============================================================
# VISUAL RETRIEVAL (NO LANGCHAIN)
# ============================================================

def retrieve_visual_candidates(query_text: str, k: int = 100, where_filter: Dict = None):
    q_emb = encode_text_clip(query_text)

    # Use Chroma's built-in filtering if provided
    res = image_collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        where=where_filter
    )

    if not res["ids"] or not res["ids"][0]:
        return []

    return [
        {
            "image_id": img_id,
            "visual_score": dist
        }
        for img_id, dist in zip(res["ids"][0], res["distances"][0])
    ]

# %%
# ============================================================
# METADATA SCORING REFINEMENTS
# ============================================================

def adaptive_alpha(query_attrs: Dict) -> float:
    return 0.1 + 0.1 * len(query_attrs)


def refined_metadata_adjustment(meta: Dict, query_attrs: Dict) -> float:
    score = 0.0

    for attr, q_val in query_attrs.items():
        m_val = meta.get(attr)
        conf = meta.get(f"confidence_{attr}", 0.0)

        if m_val == q_val:
            score += conf
        elif conf > 0.6:
            score -= 0.3 * conf

    return score


def apply_metadata_boost(candidates: List[Dict], query_attrs: Dict, exclusions: Dict = None):
    """
    Rank candidates by combining visual similarity with metadata matching.
    HARD FILTER out excluded items completely.
    
    Args:
        candidates: List of {image_id, visual_score}
        query_attrs: Attributes to INCLUDE (boost matching items)
        exclusions: Attributes to EXCLUDE (FILTER OUT completely)
    """
    if exclusions is None:
        exclusions = {}
    
    alpha = adaptive_alpha(query_attrs)
    ranked = []

    for c in candidates:
        meta = metadata_collection.get(
            ids=[c["image_id"]],
            include=["metadatas"]
        )["metadatas"][0]

        # HARD FILTER: Skip items that match exclusions
        should_exclude = False
        for attr, excluded_value in exclusions.items():
            meta_value = meta.get(attr)
            
            # Handle "any" exclusion (e.g., "plain" means no stones at all)
            if excluded_value == "any":
                # Exclude if has ANY stone (not unknown/null)
                if meta_value and meta_value not in ["unknown", "null", ""]:
                    should_exclude = True
                    print(f"🚫 Excluding {c['image_id']}: has {attr}={meta_value} (want none)")
                    break
            # Handle specific exclusion
            elif meta_value == excluded_value:
                should_exclude = True
                print(f"🚫 Excluding {c['image_id']}: has {attr}={meta_value} (excluded)")
                break
        
        # Skip this item if it matches any exclusion
        if should_exclude:
            continue
        
        # Calculate positive boost from matching attributes
        adjust = refined_metadata_adjustment(meta, query_attrs)
        
        # Final score: visual + metadata boost (no exclusion penalty needed)
        final_score = c["visual_score"] - alpha * adjust

        ranked.append({
            "image_id": c["image_id"],
            "visual_score": c["visual_score"],
            "metadata_boost": adjust,
            "final_score": final_score
        })

    return sorted(ranked, key=lambda x: x["final_score"])


def rerank_with_cross_encoder(
    query: str,
    candidates: List[Dict],
    top_k: int = 12
) -> List[Dict]:
    """
    Re-rank candidates using cross-encoder for better semantic matching.
    
    Two-stage pipeline:
    1. CLIP bi-encoder: Fast retrieval (already done)
    2. Cross-encoder: Accurate semantic re-ranking
    
    Args:
        query: User query text
        candidates: List of {image_id, visual_score, metadata_boost, ...}
        top_k: Number of results to return
    
    Returns:
        Re-ranked list of top K candidates
    """
    if not candidates:
        return []
    
    # Prepare query-document pairs for cross-encoder
    pairs = []
    for c in candidates:
        # Get BLIP caption for this image
        caption = BLIP_CAPTIONS.get(c["image_id"], "")
        
        # Get metadata
        meta = metadata_collection.get(
            ids=[c["image_id"]],
            include=["metadatas"]
        )["metadatas"][0]
        
        # Create rich text representation combining caption + metadata
        doc_text = f"{caption}. Category: {meta.get('category', 'unknown')}, Metal: {meta.get('metal', 'unknown')}, Stone: {meta.get('primary_stone', 'unknown')}"
        
        pairs.append([query, doc_text])
    
    # Score all pairs with cross-encoder (batch processing)
    print(f"🔄 Cross-encoder scoring {len(pairs)} candidates...")
    encoder = get_cross_encoder()
    cross_scores = encoder.predict(pairs, batch_size=32)
    
    # Combine scores: visual + metadata + cross-encoder
    for i, c in enumerate(candidates):
        c["cross_encoder_score"] = float(cross_scores[i])
        
        # Final score combines all signals
        # - Visual similarity (CLIP): 30% weight
        # - Metadata match: 20% weight  
        # - Semantic similarity (cross-encoder): 50% weight (highest)
        c["final_score_reranked"] = (
            -c["visual_score"] * 0.3 +  # Negate because lower distance = better
            c.get("metadata_boost", 0) * 0.2 +
            c["cross_encoder_score"] * 0.5
        )
    
    # Sort by final score (higher is better)
    ranked = sorted(candidates, key=lambda x: x["final_score_reranked"], reverse=True)
    
    print(f"✅ Re-ranked {len(ranked)} candidates, returning top {top_k}")
    return ranked[:top_k]


# %%
# ============================================================
# IMAGE UPLOAD & OCR HELPER FUNCTIONS
# ============================================================

def encode_uploaded_image(image_bytes: bytes) -> np.ndarray:
    """Encode uploaded image using CLIP model"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (max 512x512 for efficiency)
        max_size = 512
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Preprocess for CLIP
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
        
        preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                     (0.26862954, 0.26130258, 0.27577711))
        ])
        
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        
        # Encode with CLIP
        model = get_clip_model()
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            result = image_features.cpu().numpy()[0]
        
        # Memory cleanup
        del image_tensor, image_features
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")


def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image using NVIDIA NeMo Retriever OCR API with GPT-4.1-Nano fallback"""
    
    # Try NVIDIA OCR first if key is configured
    extracted_text = ""
    nvidia_failed = False
    
    if NVIDIA_API_KEY:
        try:
            # Encode image to base64
            image_b64 = base64.b64encode(image_bytes).decode()
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": [
                    {
                        "type": "image_url",
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                ]
            }
            
            # Call NVIDIA OCR API
            print(f"📞 Calling NVIDIA OCR API...")
            response = requests.post(
                NVIDIA_OCR_URL,
                headers=headers,
                json=payload,
                timeout=15  # Shorter timeout to fail fast
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Format 1: Text detections array
                if "data" in result and isinstance(result["data"], list) and len(result["data"]) > 0:
                    for data_item in result["data"]:
                        if isinstance(data_item, dict) and "text_detections" in data_item:
                            for detection in data_item["text_detections"]:
                                if "text_prediction" in detection and "text" in detection["text_prediction"]:
                                    extracted_text += detection["text_prediction"]["text"] + " "
                        elif isinstance(data_item, dict) and "content" in data_item:
                            extracted_text += data_item["content"] + " "
                
                # Format 2: Direct text field
                elif "text" in result:
                    extracted_text = result["text"]
                
                # Format 3: Choices/Results
                elif "choices" in result and len(result["choices"]) > 0:
                    if "text" in result["choices"][0]:
                        extracted_text = result["choices"][0]["text"]
                    elif "message" in result["choices"][0]:
                        extracted_text = result["choices"][0]["message"].get("content", "")
                
                extracted_text = extracted_text.strip()
                if extracted_text:
                    print(f"✅ Extracted text (NVIDIA): '{extracted_text}'")
                    return extracted_text
            
            print(f"⚠️ NVIDIA OCR failed with status {response.status_code}. Trying fallback...")
            nvidia_failed = True
            
        except Exception as e:
            print(f"⚠️ NVIDIA OCR exception: {e}. Trying fallback...")
            nvidia_failed = True
    else:
        print("ℹ️ NVIDIA_API_KEY not set. Using fallback directly.")
        nvidia_failed = True
    
    # FALLBACK: Custom GPT-4.1-Nano OCR (OpenAI Compatible)
    try:
        if not OPENAI_API_KEY:
             raise HTTPException(status_code=500, detail="OCR unavailable: Primary failed and OPENAI_API_KEY not set for fallback.")
             
        print("🔄 Using Global GPT-4.1-Nano Fallback...")
        
        # Initialize OpenAI client with custom base URL
        from openai import OpenAI
        
        client = OpenAI(
            base_url="https://apidev.navigatelabsai.com/v1",
            api_key=OPENAI_API_KEY
        )
        
        image_b64 = base64.b64encode(image_bytes).decode()
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe the handwritten text in this image exactly as it appears. Output ONLY the text, nothing else."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        extracted_text = response.choices[0].message.content.strip()
        
        if not extracted_text:
             raise HTTPException(status_code=400, detail="No readable text found in image (Fallback).")
             
        print(f"✅ Extracted text (Fallback GPT): '{extracted_text}'")
        return extracted_text
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Fallback OCR failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR failed permanently: {str(e)}")


# %%
# ============================================================
# LLM-POWERED EXPLANATION (GROQ LLAMA 3.1) - BATCH PROCESSING
# ============================================================

def batch_generate_explanations(results: List[Dict], query_attrs: Dict, user_query: str) -> List[str]:
    """Generate diverse, LLM-powered explanations for all search results in ONE API call"""
    
    if not results:
        return []
    
    # Build context for all items (handle up to 20 items in one call)
    items_context = []
    for idx, r in enumerate(results, 1):
        meta = metadata_collection.get(
            ids=[r["image_id"]],
            include=["metadatas"]
        )["metadatas"][0]
        
        matched_attrs = [v for k, v in query_attrs.items() if meta.get(k) == v]
        
        # Compact format to save tokens
        item_info = f"{idx}. {meta.get('category', 'item')} | {meta.get('metal', '?')} | {meta.get('primary_stone', '?')} | score:{r['visual_score']:.2f} | matched:{','.join(matched_attrs) if matched_attrs else 'none'}"
        items_context.append(item_info)
    
    # Compact prompt to fit more items
    prompt = f"""Query: "{user_query}"

Write 1 brief sentence for EACH item:

{chr(10).join(items_context)}

Format:
1. [sentence]
2. [sentence]
etc."""

    if groq_client is None:
        explanations = []
    else:
        try:
            # Single API call for ALL items
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Write brief jewellery recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=min(800, len(results) * 60),  # Increased for 12+ items
                top_p=0.9
            )

            # Parse response
            full_response = response.choices[0].message.content.strip()
            explanations = []

            import re
            pattern = r'^\s*(\d+)[\.:)\-]\s*(.+?)(?=^\s*\d+[\.:)\-]|\Z)'
            matches = re.findall(pattern, full_response, re.MULTILINE | re.DOTALL)

            if matches and len(matches) >= len(results):
                for num, text in matches[:len(results)]:
                    clean_text = ' '.join(text.strip().split())
                    if clean_text and len(clean_text) > 10:
                        explanations.append(clean_text)

            if len(explanations) >= len(results):
                return explanations[:len(results)]

            # If incomplete, pad with fallback
            print(f"⚠️ LLM returned {len(explanations)}/{len(results)} explanations, padding with fallback")

        except Exception as e:
            print(f"⚠️ LLM explanation failed: {e}, using fallback")
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
        
        if matched_attrs and r['visual_score'] < 1.3:
            explanations.append(
                f"Excellent {category} featuring {' and '.join(matched_attrs)}. High visual similarity (score: {r['visual_score']:.2f})."
            )
        elif matched_attrs:
            explanations.append(
                f"Beautiful {metal} {category} with {stone}. Features {' and '.join(matched_attrs)}."
            )
        elif r['visual_score'] < 1.3:
            explanations.append(
                f"Highly similar {category} with excellent visual match. {metal.capitalize()} with {stone}."
            )
        else:
            explanations.append(
                f"Recommended {metal} {category} with {stone}. Good visual similarity."
            )
    
    return explanations
# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health")
def health_check():
    """Health check endpoint for HF Spaces monitoring"""
    return {
        "status": "healthy",
        "models_loaded": {
            "clip": clip_model is not None,
            "cross_encoder": cross_encoder is not None,
            "blip_captions": len(BLIP_CAPTIONS) > 0
        },
        "database": {
            "images": image_collection.count(),
            "metadata": metadata_collection.count()
        }
    }

@app.post("/search/text")
def search_text(req: TextSearchRequest):
    # Detect intent from text
    if req.query.strip():
        intent = detect_intent_and_attributes(req.query)
        attrs = intent["attributes"]
    else:
        intent = {"intent": "filter", "attributes": {}, "exclusions": {}}
        attrs = {}

    # === DUAL-STAGE FILTERING STRATEGY ===
    # 1. Identify valid IDs from Metadata Collection (Source of Truth)
    # 2. Use those IDs to filter Vector Search results

    # Construct WHERE clause for Metadata Collection
    where_clauses = []
    
    if req.filters:
        for key, value in req.filters.items():
            where_clauses.append({key: value.lower()}) # Explicit filters
    
    # Also apply attributes detected from text as generic filters if user didn't specify explicit ones
    # (Optional: this makes "emerald ring" implies primary_stone=emerald)
    # But usually we let visual search handle text unless it's strict.
    
    final_where = None
    if len(where_clauses) > 1:
        final_where = {"$and": where_clauses}
    elif len(where_clauses) == 1:
        final_where = where_clauses[0]
    
    valid_ids = None
    if final_where:
        # Fetch ALL valid IDs matching the filter
        print(f"🔍 Filtering metadata with: {final_where}")
        meta_res = metadata_collection.get(where=final_where, include=["metadatas"])
        if meta_res["ids"]:
            valid_ids = set(meta_res["ids"])
            print(f"✅ Found {len(valid_ids)} valid items matching filters.")
        else:
            print("⚠️ No items match the filters.")
            return {"query": req.query, "intent": attrs, "results": []}

    # === EXECUTE SEARCH ===
    
    # Case A: Filter Only (No Text Query)
    if not req.query.strip() and valid_ids:
        # Just return the matching items (Top K)
        candidates = [{"image_id": vid, "visual_score": 0.0} for vid in list(valid_ids)[:req.top_k]]
        ranked = candidates # No ranking needed without text
        explanations = ["Filtered result"] * len(ranked)
        
    # Case B: Text Query (with or without Filter)
    else:
        search_query = req.query if req.query.strip() else "jewellery"
        
        # We perform a BROADER vector search, then filter in Python
        # Retrieve K*5 or at least 100 to ensure we find intersections
        fetch_k = 200 if valid_ids else 40 
        
        # Note: We do NOT pass 'where' to retrieve_visual_candidates because 
        # image_collection lacks metadata. We filter manually.
        candidates = retrieve_visual_candidates(search_query, k=fetch_k)
        
        filtered_candidates = []
        for c in candidates:
            if valid_ids is not None:
                if c["image_id"] in valid_ids:
                    filtered_candidates.append(c)
            else:
                filtered_candidates.append(c)
        
        # Apply strict limit now
        filtered = filtered_candidates # apply_metadata_boost(filtered_candidates, attrs, {})
        
        # Cross-encoder re-ranking
        if req.use_reranking and filtered and req.query.strip():
            ranked = rerank_with_cross_encoder(req.query, filtered, req.top_k)
        else:
            ranked = filtered[:req.top_k]
            
        # Explanations
        if req.use_explanations and req.query.strip():
            explanations = batch_generate_explanations(ranked, attrs, search_query)
        else:
            explanations = ["Match found"] * len(ranked)

    # === FORMAT RESULTS ===
    results = []
    
    # Fetch metadata for final results
    if ranked:
        ranked_ids = [r["image_id"] for r in ranked]
        metas = metadata_collection.get(ids=ranked_ids, include=["metadatas"])["metadatas"]
        meta_map = {rid: m for rid, m in zip(ranked_ids, metas)}
    else:
        meta_map = {}

    for r, explanation in zip(ranked, explanations):
        results.append({
            "image_id": r["image_id"],
            "explanation": explanation,
            "metadata": meta_map.get(r["image_id"], {}),
            "scores": {
                "visual": r["visual_score"],
                "final": r.get("visual_score", 0) # simplified
            }
        })

    return {
        "query": req.query,
        "intent": attrs,
        "results": results
    }

    return {
        "query": req.query,
        "intent": attrs,
        "results": results
    }

# %%
@app.post("/search/similar")
def search_similar(req: SimilarSearchRequest):
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
            "visual_score": dist
        }
        for img_id, dist in zip(res["ids"][0], res["distances"][0])
        if img_id != req.image_id
    ]

    ranked = apply_metadata_boost(candidates, attrs, {})[:req.top_k]

    # Generate all explanations in one batch LLM call
    # For similar search, use the base image ID as the query context
    query_context = f"items similar to {req.image_id}"
    explanations = batch_generate_explanations(ranked, attrs, query_context)
    
    results = []
    for r, explanation in zip(ranked, explanations):
        results.append({
            "image_id": r["image_id"],
            "explanation": explanation,
            "scores": {
                "visual": r["visual_score"],
                "metadata": r["metadata_boost"],
                "final": r["final_score"]
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
                "visual_score": dist
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
        
        # Apply metadata boost (no exclusions for image upload)
        ranked = apply_metadata_boost(candidates, attrs, {})[:top_k]
        
        # Generate explanations in batch
        query_context = f"items visually similar to uploaded image"
        explanations = batch_generate_explanations(ranked, attrs, query_context)
        
        results = []
        for r, explanation in zip(ranked, explanations):
            results.append({
                "image_id": r["image_id"],
                "explanation": explanation,
                "scores": {
                    "visual": r["visual_score"],
                    "metadata": r["metadata_boost"],
                    "final": r["final_score"]
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
    top_k: int = 12
):
    """
    Extract text from uploaded image using NVIDIA NeMo OCR,
    then perform text-based search with the extracted query.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Extract text using NVIDIA OCR
        extracted_text = extract_text_from_image(image_bytes)
        
        print(f"📝 Extracted text from image: '{extracted_text}'")
        
        # Use the extracted text for normal text search
        intent = detect_intent_and_attributes(extracted_text)
        attrs = intent["attributes"]
        exclusions = intent.get("exclusions", {})
        
        # Stage 1: CLIP retrieval (reduced to k=40 for HF Spaces)
        candidates = retrieve_visual_candidates(extracted_text, k=40)
        
        # Stage 2: Metadata boost + exclusion filtering
        filtered = apply_metadata_boost(candidates, attrs, exclusions)
        
        # Stage 3: Cross-encoder re-ranking
        ranked = rerank_with_cross_encoder(extracted_text, filtered, top_k)
        
        # Generate explanations in batch
        explanations = batch_generate_explanations(ranked, attrs, extracted_text)
        
        results = []
        for r, explanation in zip(ranked, explanations):
            results.append({
                "image_id": r["image_id"],
                "explanation": explanation,
                "scores": {
                    "visual": r["visual_score"],
                    "metadata": r["metadata_boost"],
                    "final": r["final_score"]
                }
            })
        
        return {
            "query_type": "ocr_extracted",
            "extracted_text": extracted_text,
            "intent": attrs,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR search failed: {str(e)}")

# %%
@app.get("/image/{image_id}")
def get_image(image_id: str):
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
