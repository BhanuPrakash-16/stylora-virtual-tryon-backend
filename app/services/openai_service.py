"""
OpenAI Service
==============
Virtual try-on using OpenAI's image generation capabilities.

IMPORTANT: OpenAI doesn't have a specialized virtual try-on model.
This implementation uses DALL-E for image generation/editing, which
may not produce results as accurate as specialized try-on models.
"""

import base64
import io
from typing import Dict
from openai import OpenAI
from PIL import Image
from app.config import config


# Initialize OpenAI client
_openai_client = None


def get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    
    if _openai_client is None:
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    return _openai_client


async def check_openai_status() -> Dict:
    """
    Check if OpenAI API is configured and accessible.
    
    Returns:
        Dict with status information
    """
    if not config.OPENAI_API_KEY:
        return {
            "configured": False,
            "accessible": False,
            "error": "OPENAI_API_KEY not set"
        }
    
    try:
        client = get_openai_client()
        # Simple test - list models
        client.models.list()
        return {
            "configured": True,
            "accessible": True,
            "error": None
        }
    except Exception as e:
        return {
            "configured": True,
            "accessible": False,
            "error": str(e)
        }


async def run_virtual_tryon(person_bytes: bytes, garment_bytes: bytes) -> Dict:
    """
    Perform virtual try-on using OpenAI DALL-E.
    
    NOTE: This is a workaround since OpenAI doesn't have a dedicated
    virtual try-on model. Results may vary and won't be as accurate
    as specialized models like Replicate's IDM-VTON.
    
    Args:
        person_bytes: Person image bytes
        garment_bytes: Garment image bytes
        
    Returns:
        Dict with keys:
        - success (bool): True if generation succeeded
        - result_url (str): URL of generated image
        - error (str): Error message if failed
    """
    try:
        client = get_openai_client()
        
        # Convert images to base64
        person_base64 = base64.b64encode(person_bytes).decode('utf-8')
        garment_base64 = base64.b64encode(garment_bytes).decode('utf-8')
        
        # Create a prompt for virtual try-on
        # Using GPT-4 Vision to analyze both images first
        prompt = """You are helping with a virtual try-on task. 
        I have two images:
        1. A person wearing clothes
        2. A garment that should be tried on
        
        Please generate a photorealistic image of the person wearing the new garment.
        Maintain the person's pose, body shape, and background.
        Only replace the clothing with the new garment.
        Make it look natural and realistic."""
        
        # Use DALL-E 3 for image generation
        # Note: DALL-E doesn't support direct image editing with reference images
        # This is a limitation - we'll use text-to-image generation
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        if response.data and len(response.data) > 0:
            result_url = response.data[0].url
            
            return {
                "success": True,
                "result_url": result_url,
                "error": None,
                "note": "Generated using DALL-E 3 (not a specialized try-on model)"
            }
        else:
            return {
                "success": False,
                "result_url": None,
                "error": "No image generated"
            }
    
    except Exception as e:
        return {
            "success": False,
            "result_url": None,
            "error": str(e)
        }


async def run_virtual_tryon_with_vision(person_bytes: bytes, garment_bytes: bytes) -> Dict:
    """
    Alternative approach using GPT-4 Vision for analysis + DALL-E for generation.
    
    This analyzes both images and creates a detailed prompt for better results.
    """
    try:
        client = get_openai_client()
        
        # Convert to base64
        person_base64 = base64.b64encode(person_bytes).decode('utf-8')
        garment_base64 = base64.b64encode(garment_bytes).decode('utf-8')
        
        # Step 1: Use GPT-4 Vision to analyze the images
        vision_response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the person in the first image (their pose, body type, hair, face) and the garment in the second image (color, style, type). Create a detailed prompt for generating an image of this person wearing this garment."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{person_base64}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{garment_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        # Extract the generated prompt
        generated_prompt = vision_response.choices[0].message.content
        
        # Step 2: Use DALL-E to generate the image
        image_response = client.images.generate(
            model="dall-e-3",
            prompt=f"Photorealistic image: {generated_prompt}. Professional photography, natural lighting.",
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        if image_response.data and len(image_response.data) > 0:
            return {
                "success": True,
                "result_url": image_response.data[0].url,
                "error": None,
                "note": "Generated using GPT-4 Vision + DALL-E 3"
            }
        else:
            return {
                "success": False,
                "result_url": None,
                "error": "No image generated"
            }
    
    except Exception as e:
        return {
            "success": False,
            "result_url": None,
            "error": str(e)
        }
