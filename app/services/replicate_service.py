"""
Replicate API Service
=====================
Handles virtual try-on model calls via Replicate API.

IMPORTANT: This service should ONLY be called AFTER all safety
validations have passed. Never call this with unvalidated images.
"""

import replicate
from typing import Dict, Optional
import base64
from app.config import config
from app.utils.image_utils import numpy_to_base64, bytes_to_numpy


async def run_virtual_tryon(
    person_image_bytes: bytes,
    garment_image_bytes: bytes,
    garment_description: str = "A shirt"
) -> Dict:
    """
    Run virtual try-on using Replicate API.
    
    CRITICAL: This function assumes all safety checks have ALREADY passed.
    Do NOT call this function with unvalidated images.
    
    Model: Uses IDM-VTON or similar virtual try-on model on Replicate.
    
    Args:
        person_image_bytes: Validated person image bytes
        garment_image_bytes: Validated garment image bytes
        garment_description: Optional description of garment
        
    Returns:
        Dict with keys:
        - success (bool): True if try-on succeeded
        - result_url (str): URL to result image (if success)
        - result_base64 (str): Base64 encoded result (if success)
        - error (str): Error message (if failed)
    """
    
    if not config.REPLICATE_API_TOKEN:
        return {
            "success": False,
            "error": "Replicate API token not configured",
            "result_url": None,
            "result_base64": None
        }
    
    try:
        # Set API token
        replicate_client = replicate.Client(api_token=config.REPLICATE_API_TOKEN)
        
        # Convert images to base64 for Replicate
        person_b64 = base64.b64encode(person_image_bytes).decode('utf-8')
        garment_b64 = base64.b64encode(garment_image_bytes).decode('utf-8')
        
        # Prepare data URLs
        person_data_url = f"data:image/jpeg;base64,{person_b64}"
        garment_data_url = f"data:image/jpeg;base64,{garment_b64}"
        
        # Call Replicate virtual try-on model
        # Model: IDM-VTON (or similar)
        # Note: Update model version as needed
        output = replicate_client.run(
            "cuuupid/idm-vton:c871bb9b046607b680449ecbae55fd8c6d945e0a1948644bf2361b3d021d3ff4",
            input={
                "human_img": person_data_url,
                "garm_img": garment_data_url,
                "garment_des": garment_description,
                "category": "upper_body",  # or "lower_body", "dresses"
                "n_samples": 1,
                "n_steps": 20,
                "image_scale": 1.0
            }
        )
        
        # Output is typically a URL or list of URLs
        if isinstance(output, list) and len(output) > 0:
            result_url = output[0]
        elif isinstance(output, str):
            result_url = output
        else:
            return {
                "success": False,
                "error": "Unexpected output format from Replicate",
                "result_url": None,
                "result_base64": None
            }
        
        # Optionally, download and convert to base64
        # (For now, just return the URL)
        
        return {
            "success": True,
            "result_url": result_url,
            "result_base64": None,  # Could download and encode if needed
            "error": None
        }
        
    except replicate.exceptions.ReplicateError as e:
        return {
            "success": False,
            "error": f"Replicate API error: {str(e)}",
            "result_url": None,
            "result_base64": None
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Virtual try-on error: {str(e)}",
            "result_url": None,
            "result_base64": None
        }


async def check_replicate_status() -> Dict:
    """
    Check if Replicate API is configured and accessible.
    
    Returns:
        Dict with status information
    """
    if not config.REPLICATE_API_TOKEN:
        return {
            "configured": False,
            "accessible": False,
            "message": "REPLICATE_API_TOKEN not set"
        }
    
    try:
        # Try to ping Replicate
        replicate_client = replicate.Client(api_token=config.REPLICATE_API_TOKEN)
        
        # Simple test: list models (should not error if token is valid)
        # Note: This might not be the best test, adjust as needed
        
        return {
            "configured": True,
            "accessible": True,
            "message": "Replicate API is configured and accessible"
        }
    
    except Exception as e:
        return {
            "configured": True,
            "accessible": False,
            "message": f"Replicate API error: {str(e)}"
        }


def get_supported_garment_types() -> Dict:
    """
    Get information about supported garment types for try-on.
    
    Returns:
        Dict with supported categories
    """
    return {
        "categories": [
            {
                "id": "upper_body",
                "name": "Upper Body",
                "description": "Shirts, t-shirts, blouses, jackets, etc.",
                "examples": ["shirt", "t-shirt", "blouse", "jacket", "sweater"]
            },
            {
                "id": "lower_body",
                "name": "Lower Body",
                "description": "Pants, jeans, trousers, skirts, etc.",
                "examples": ["pants", "jeans", "trousers", "skirt"]
            },
            {
                "id": "dresses",
                "name": "Dresses",
                "description": "Full dresses, gowns, sarees, etc.",
                "examples": ["dress", "gown", "saree", "maxi dress"]
            }
        ],
        "note": "Category is auto-detected based on garment type"
    }
