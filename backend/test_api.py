import requests
import json

BASE_URL = "http://localhost:8000"

def search(query, filters=None):
    payload = {
        "query": query,
        "filters": filters or {},
        "top_k": 5,
        "use_reranking": False,
        "use_explanations": False
    }
    try:
        response = requests.post(f"{BASE_URL}/search/text", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

def run_tests():
    print("=== Test 1: Filter Only (Emerald) ===")
    res1 = search("", {"primary_stone": "Emerald"})
    if res1 and res1["results"]:
        print(f"Success. Found {len(res1['results'])} items.")
        print(f"Sample: {res1['results'][0]['metadata'].get('primary_stone')}")
    else:
        print("Failed. No results or error.")

    print("\n=== Test 2: Filter (Gold) + Query (Necklace) ===")
    res2 = search("necklace", {"metal": "Gold"})
    if res2 and res2["results"]:
        print(f"Success. Found {len(res2['results'])} items.")
        print(f"Sample Metal: {res2['results'][0]['metadata'].get('metal')}")
        print(f"Sample Category: {res2['results'][0]['metadata'].get('category')}")
    else:
        print("Failed. No results.")

    print("\n=== Test 3: Filter (Ring) + Query (Blue) ===")
    res3 = search("blue", {"category": "Ring"})
    if res3 and res3["results"]:
        print(f"Success. Found {len(res3['results'])} items.")
        print(f"Sample Category: {res3['results'][0]['metadata'].get('category')}")
    else:
        print("Failed. No results.")
        
if __name__ == "__main__":
    run_tests()
