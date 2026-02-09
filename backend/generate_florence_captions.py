#!/usr/bin/env python3
"""
Generate Florence-2 captions for all jewelry images using CUDA acceleration.
This script uses the Florence-2-base model for faster processing.
"""

import json
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
import gc

# Configuration
IMAGES_DIR = "backend/data/tanishq/images"
OUTPUT_FILE = "data/tanishq/florence_captions_all.json"
MODEL_NAME = "microsoft/Florence-2-base"  # Using base model for faster processing
BATCH_SIZE = 4  # Process multiple images at once
DEVICE = "cuda"

def load_model():
    """Load Florence-2 model and processor with CUDA optimization."""
    print(f"🚀 Checking CUDA availability...")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"\n📥 Loading Florence-2-base model on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # Use FP16 for faster inference
        trust_remote_code=True,
        attn_implementation="eager"  # Use eager attention to avoid SDPA issues
    ).to(DEVICE)
    model.eval()  # Set to evaluation mode
    print("✅ Model loaded successfully on CUDA!")
    return processor, model

def generate_caption_batch(image_paths, processor, model):
    """Generate captions for a batch of images."""
    try:
        # Load and process images
        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        
        # Prepare inputs for detailed caption
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = processor(
            text=[prompt] * len(images),
            images=images,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        
        # Generate captions
        with torch.no_grad(), torch.cuda.amp.autocast():  # Use automatic mixed precision
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,  # Reduced for faster processing
                num_beams=3,
                do_sample=False,
                early_stopping=True
            )
        
        # Decode captions
        captions = []
        for i, (generated_id, image) in enumerate(zip(generated_ids, images)):
            generated_text = processor.batch_decode([generated_id], skip_special_tokens=False)[0]
            
            # Parse the response
            parsed_answer = processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(image.width, image.height)
            )
            
            caption = parsed_answer.get("<MORE_DETAILED_CAPTION>", "")
            captions.append(caption)
        
        # Clear memory
        del inputs, generated_ids
        torch.cuda.empty_cache()
        
        return captions
        
    except Exception as e:
        print(f"Error processing batch: {e}")
        return [None] * len(image_paths)

def main():
    """Main function to generate captions for all images."""
    # Verify CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA is not available! Please check your PyTorch installation.")
        return
    
    # Load model
    processor, model = load_model()
    
    # Get all image files
    images_path = Path(IMAGES_DIR)
    image_files = sorted([
        f for f in images_path.glob("*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])
    
    print(f"\n📊 Found {len(image_files)} images to process")
    print(f"⚡ Processing in batches of {BATCH_SIZE}")
    
    # Generate captions in batches
    captions_data = []
    
    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="🎨 Generating captions"):
        batch_files = image_files[i:i + BATCH_SIZE]
        batch_captions = generate_caption_batch(batch_files, processor, model)
        
        for image_file, caption in zip(batch_files, batch_captions):
            if caption:
                captions_data.append({
                    "image": image_file.name,
                    "caption": caption
                })
        
        # Periodic garbage collection
        if i % (BATCH_SIZE * 10) == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Save captions
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(captions_data, f, indent=2)
    
    print(f"\n✅ Generated {len(captions_data)} captions")
    print(f"✅ Saved to {OUTPUT_FILE}")
    
    # Final cleanup
    del model, processor
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
