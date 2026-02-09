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
        from image_helpers import encode_uploaded_image
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
        
        # Apply metadata boost
        ranked = apply_metadata_boost(candidates, attrs)[:top_k]
        
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
        from image_helpers import extract_text_from_image
        extracted_text = extract_text_from_image(image_bytes)
        
        print(f"📝 Extracted text from image: '{extracted_text}'")
        
        # Use the extracted text for normal text search
        intent = detect_intent_and_attributes(extracted_text)
        attrs = intent["attributes"]
        
        candidates = retrieve_visual_candidates(extracted_text, k=100)
        ranked = apply_metadata_boost(candidates, attrs)[:top_k]
        
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

