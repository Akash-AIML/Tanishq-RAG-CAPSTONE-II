import chromadb

client = chromadb.PersistentClient(path="./backend/chroma_primary")
meta_col = client.get_collection("jewelry_metadata")

# Get a sample of items to check their metadata
print("Fetching 10 items from jewelry_metadata...")
results = meta_col.get(limit=10, include=["metadatas"])

if results["metadatas"]:
    print(f"Sample Metadata 1: {results['metadatas'][0]}")
    
    # Check for specific keys
    found_metals = set()
    found_stones = set()
    
    # Getting ALL might be too slow if huge, so let's try to query specifically
    print("\nAttempting direct query for metal='gold'...")
    gold_res = meta_col.get(where={"metal": "gold"})
    print(f"Found {len(gold_res['ids'])} items with metal='gold'")
    
    print("\nAttempting direct query for primary_stone='emerald'...")
    emerald_res = meta_col.get(where={"primary_stone": "emerald"})

img_col = client.get_collection("jewelry_images")
print("\nFetching 5 items from jewelry_images...")
img_res = img_col.get(limit=5)
print(f"Sample Image IDs: {img_res['ids']}")

