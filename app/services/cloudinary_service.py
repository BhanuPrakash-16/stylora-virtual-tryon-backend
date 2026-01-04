"""
Cloudinary Service
==================
Upload and manage images on Cloudinary.
"""

import cloudinary
import cloudinary.uploader
from typing import Dict, Optional
import base64
from app.config import config


# Initialize Cloudinary
_cloudinary_initialized = False


def initialize_cloudinary():
    """
    Initialize Cloudinary with API credentials.
    
    This should be called once on application startup.
    """
    global _cloudinary_initialized
    
    if _cloudinary_initialized:
        return
    
    try:
        if all([config.CLOUDINARY_CLOUD_NAME, config.CLOUDINARY_API_KEY, config.CLOUDINARY_API_SECRET]):
            cloudinary.config(
                cloud_name=config.CLOUDINARY_CLOUD_NAME,
                api_key=config.CLOUDINARY_API_KEY,
                api_secret=config.CLOUDINARY_API_SECRET,
                secure=True
            )
            print("✓ Cloudinary initialized")
            _cloudinary_initialized = True
        else:
            print("⚠ Cloudinary not initialized (credentials not provided)")
            print("  Image hosting features will be disabled")
    
    except Exception as e:
        print(f"✗ Cloudinary initialization failed: {e}")


def is_cloudinary_initialized() -> bool:
    """Check if Cloudinary is initialized."""
    return _cloudinary_initialized


async def upload_image_to_cloudinary(
    image_data: bytes,
    folder: str = "stylora/tryons",
    public_id: Optional[str] = None
) -> Dict:
    """
    Upload image to Cloudinary.
    
    Args:
        image_data: Image bytes
        folder: Cloudinary folder path
        public_id: Optional public ID for the image
        
    Returns:
        Dict with keys:
        - success (bool): True if upload succeeded
        - url (str): Secure URL of uploaded image
        - public_id (str): Cloudinary public ID
        - error (str): Error message (if failed)
    """
    if not is_cloudinary_initialized():
        return {
            "success": False,
            "url": None,
            "public_id": None,
            "error": "Cloudinary not initialized"
        }
    
    try:
        # Convert bytes to base64 data URI
        base64_data = base64.b64encode(image_data).decode('utf-8')
        data_uri = f"data:image/jpeg;base64,{base64_data}"
        
        # Upload to Cloudinary
        upload_params = {
            "folder": folder,
            "resource_type": "image",
            "overwrite": False
        }
        
        if public_id:
            upload_params["public_id"] = public_id
        
        result = cloudinary.uploader.upload(data_uri, **upload_params)
        
        return {
            "success": True,
            "url": result.get("secure_url"),
            "public_id": result.get("public_id"),
            "width": result.get("width"),
            "height": result.get("height"),
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "url": None,
            "public_id": None,
            "error": str(e)
        }


async def upload_result_image(
    result_bytes: bytes,
    user_uid: Optional[str] = None
) -> Dict:
    """
    Upload try-on result image to Cloudinary.
    
    Args:
        result_bytes: Result image bytes
        user_uid: Optional user ID (for organizing uploads)
        
    Returns:
        Dict with upload result
    """
    folder = f"stylora/results/{user_uid}" if user_uid else "stylora/results/anonymous"
    
    return await upload_image_to_cloudinary(
        image_data=result_bytes,
        folder=folder
    )


async def delete_image_from_cloudinary(public_id: str) -> Dict:
    """
    Delete image from Cloudinary.
    
    Args:
        public_id: Cloudinary public ID
        
    Returns:
        Dict with deletion result
    """
    if not is_cloudinary_initialized():
        return {
            "success": False,
            "error": "Cloudinary not initialized"
        }
    
    try:
        result = cloudinary.uploader.destroy(public_id)
        
        return {
            "success": result.get("result") == "ok",
            "error": None if result.get("result") == "ok" else "Deletion failed"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
