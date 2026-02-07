import chromadb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_primary")

print(f"Connecting to Chroma at {CHROMA_PATH}")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
metadata_collection = chroma_client.get_collection("jewelry_metadata")

# Check strict retrieval
target_id = "ring_ring_051.jpg"
print(f"Checking metadata for {target_id}...")
try:
    item = metadata_collection.get(ids=[target_id])
    if item['ids']:
        print("Item Found:")
        print(item['metadatas'][0])
    else:
        print("Item NOT found in metadata collection.")
except Exception as e:
    print(f"Error fetching item: {e}")

# Check image collection metadata
print(f"Checking image collection metadata for {target_id}...")
try:
    image_collection = chroma_client.get_collection("jewelry_images")
    item = image_collection.get(ids=[target_id])
    if item['ids']:
        print("Image Collection Item Found. Metadata:")
        print(item['metadatas'][0])
    else:
        print("Item NOT found in image collection.")
except Exception as e:
    print(f"Error fetching image item: {e}")
