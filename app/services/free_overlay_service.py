"""
Stylora: FREE Overlay-Based Virtual Try-On
===========================================
100% Free, Railway-Safe, No GPU Required

This is a simple, deterministic overlay approach that:
- Works on any platform (Railway free tier safe)
- No heavy ML dependencies
- Fast and predictable results
- Perfect for MVP and hackathons

Mode: BETA (Overlay-Based Preview)
"""

from PIL import Image
import io
import base64
from typing import Dict, Tuple


def create_simple_overlay(person_bytes: bytes, garment_bytes: bytes) -> Dict:
    """
    Simple overlay-based virtual try-on (100% FREE).
    
    This is a basic geometric overlay approach that:
    1. Loads the person and garment images
    2. Resizes garment to fit person proportions
    3. Overlays garment at estimated position
    4. Returns result as base64
    
    Args:
        person_bytes: Person image bytes
        garment_bytes: Garment image bytes
        
    Returns:
        Dict with:
        - success (bool): True if successful
        - result_url (str): data:image URL
        - result_base64 (str): base64 encoded result
        - mode (str): "beta_overlay"
        - note (str): Beta disclaimer
        - error (str): Error message if failed
    """
    try:
        # Load images
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGBA")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGBA")
        
        # Get person dimensions
        person_width, person_height = person_img.size
        
        # Resize garment to fit person (proportional sizing)
        # Garment covers approx 60% width, 50% height of person
        garment_target_width = int(person_width * 0.6)
        garment_target_height = int(person_height * 0.5)
        
        # Maintain aspect ratio
        garment_aspect = garment_img.width / garment_img.height
        target_aspect = garment_target_width / garment_target_height
        
        if garment_aspect > target_aspect:
            # Garment is wider - fit to width
            final_width = garment_target_width
            final_height = int(final_width / garment_aspect)
        else:
            # Garment is taller - fit to height
            final_height = garment_target_height
            final_width = int(final_height * garment_aspect)
        
        garment_resized = garment_img.resize((final_width, final_height), Image.Resampling.LANCZOS)
        
        # Position garment on person
        # Center horizontally, position vertically at chest area (25% from top)
        x_position = (person_width - final_width) // 2
        y_position = int(person_height * 0.25)
        
        # Create result by pasting garment over person
        result = person_img.copy()
        result.paste(garment_resized, (x_position, y_position), garment_resized)
        
        # Convert to RGB for JPEG
        result_rgb = result.convert("RGB")
        
        # Save to bytes
        output_buffer = io.BytesIO()
        result_rgb.save(output_buffer, format="JPEG", quality=90)
        output_buffer.seek(0)
        result_bytes = output_buffer.read()
        
        # Convert to base64
        result_base64 = base64.b64encode(result_bytes).decode('utf-8')
        result_data_url = f"data:image/jpeg;base64,{result_base64}"
        
        return {
            "success": True,
            "result_url": result_data_url,
            "result_base64": result_base64,
            "mode": "beta_overlay",
            "note": "This is a beta preview using simple overlay. Results may not be perfectly accurate.",
            "error": None,
            "processing_time": "instant",
            "cost": "free"
        }
        
    except Exception as e:
        return {
            "success": False,
            "result_url": None,
            "result_base64": None,
           "mode": "beta_overlay",
            "note": "Processing failed",
            "error": str(e),
            "processing_time": None,
            "cost": "free"
        }


def create_overlay_composite(person_bytes: bytes, garment_bytes: bytes, opacity: float = 0.9) -> Dict:
    """
    Alternative overlay with transparency control.
    
    Args:
        person_bytes: Person image bytes
        garment_bytes: Garment image bytes
        opacity: Garment opacity (0.0-1.0)
        
    Returns:
        Same format as create_simple_overlay
    """
    try:
        # Load images
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGBA")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGBA")
        
        # Apply opacity to garment
        garment_with_opacity = garment_img.copy()
        alpha = garment_with_opacity.split()[3]
        alpha = alpha.point(lambda p: int(p * opacity))
        garment_with_opacity.putalpha(alpha)
        
        # Resize and position (same as simple overlay)
        person_width, person_height = person_img.size
        garment_target_width = int(person_width * 0.6)
        garment_target_height = int(person_height * 0.5)
        
        garment_aspect = garment_img.width / garment_img.height
        target_aspect = garment_target_width / garment_target_height
        
        if garment_aspect > target_aspect:
            final_width = garment_target_width
            final_height = int(final_width / garment_aspect)
        else:
            final_height = garment_target_height
            final_width = int(final_height * garment_aspect)
        
        garment_resized = garment_with_opacity.resize((final_width, final_height), Image.Resampling.LANCZOS)
        
        x_position = (person_width - final_width) // 2
        y_position = int(person_height * 0.25)
        
        # Composite with transparency
        result = person_img.copy()
        result.paste(garment_resized, (x_position, y_position), garment_resized)
        result_rgb = result.convert("RGB")
        
        # Save
        output_buffer = io.BytesIO()
        result_rgb.save(output_buffer, format="JPEG", quality=90)
        output_buffer.seek(0)
        result_bytes = output_buffer.read()
        
        result_base64 = base64.b64encode(result_bytes).decode('utf-8')
        result_data_url = f"data:image/jpeg;base64,{result_base64}"
        
        return {
            "success": True,
            "result_url": result_data_url,
            "result_base64": result_base64,
            "mode": "beta_overlay_transparent",
            "note": "Beta preview with transparency control. Results may not be perfectly accurate.",
            "error": None,
            "opacity": opacity,
            "processing_time": "instant",
            "cost": "free"
        }
        
    except Exception as e:
        return {
            "success": False,
            "result_url": None,
            "result_base64": None,
            "mode": "beta_overlay_transparent",
            "note": "Processing failed",
            "error": str(e),
            "processing_time": None,
            "cost": "free"
        }
