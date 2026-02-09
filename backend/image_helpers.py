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
        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")


def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image using NVIDIA NeMo Retriever OCR API"""
    
    if not NVIDIA_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="NVIDIA_API_KEY not configured. Please add it to your .env file."
        )
    
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
        response = requests.post(
            NVIDIA_OCR_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"NVIDIA OCR API error: {response.text}"
            )
        
        # Extract text from response
        result = response.json()
        
        # Parse OCR response - extract all text content
        extracted_text = ""
        if "data" in result and len(result["data"]) > 0:
            for item in result["data"]:
                if "content" in item:
                    extracted_text += item["content"] + " "
        
        extracted_text = extracted_text.strip()
        
        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the image. Please ensure the image contains clear, readable text."
            )
        
        return extracted_text
        
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="OCR request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"OCR request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing error: {str(e)}")

