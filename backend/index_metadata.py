import json
import clip
import torch
import chromadb
import os

# =========================
# CONFIG
# =========================
BASE_DIR = "/home/akash/Jewellary_RAG"
DATA_DIR = os.path.join(BASE_DIR, "data/tanishq")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_primary")
METADATA_PATH = os.path.join(DATA_DIR, "generated_metadata.json")

# =========================
# LOAD METADATA JSON
# =========================

print(f"📂 Loading metadata from {METADATA_PATH}...")
with open(METADATA_PATH) as f:
    metadata_json = json.load(f)

# =========================
# LOAD CLIP TEXT ENCODER
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Loading CLIP model on {device}...")

model, _ = clip.load("ViT-B/16", device=device)
model.eval()

# =========================
# INIT CHROMA
# =========================

print(f"🔹 Connecting to Chroma DB at {CHROMA_PATH}...")
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Use get_or_create_collection to avoid errors if it exists
metadata_collection = client.get_or_create_collection(
    name="jewelry_metadata"
)

# =========================
# EMBED METADATA TEXT
# =========================

ids = []
embeddings = []
documents = []
metadatas = []

print("🔹 Encoding metadata text...")
count = 0
for img_id, meta in metadata_json.items():

    text = meta["metadata_text"]

    # Tokenize and encode
    tokens = clip.tokenize([text]).to(device)

    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    embedding = emb.cpu().numpy()[0]

    ids.append(img_id)
    embeddings.append(embedding)
    documents.append(text)
    metadatas.append(meta)
    
    count += 1
    if count % 100 == 0:
        print(f"   Processed {count} items...")

# =========================
# INSERT INTO CHROMA
# =========================

print(f"💾 Indexing {len(ids)} items into ChromaDB...")
# Use upsert to handle existing IDs gracefully (update/insert)
metadata_collection.upsert(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas
)

print(f"✅ indexed {len(ids)} items successfully")
