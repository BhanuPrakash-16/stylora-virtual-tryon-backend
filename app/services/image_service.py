"""
Image Service
=============
Simplified image validation for testing.
"""

import io
from PIL import Image
from typing import Dict


def validate_and_prepare_image(image_bytes: bytes, max_size: int = 1024) -> Dict:
    """
    Validate image format (SIMPLIFIED FOR TESTING).
    
    Returns:
        Dict with 'valid' (bool) and 'error' (str) keys
    """
    try:
        # Try to open image with PIL
        img = Image.open(io.BytesIO(image_bytes))
        
        # Get dimensions
        width, height = img.size
        
        # Very basic checks
        if width < 50 or height < 50:
            return {
                "valid": False,
                "image": None,
                "error": "Image too small (minimum 50x50 pixels)",
                "dimensions": (height, width)
            }
        
        # All good!
        return {
            "valid": True,
            "image": None,  # Not needed for Replicate
            "error": None,
            "dimensions": (height, width)
        }
    
    except Exception as e:
        return {
            "valid": False,
            "image": None,
            "error": f"Invalid image file: {str(e)}",
            "dimensions": None
        }
