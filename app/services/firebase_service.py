"""
Firebase Admin SDK Service
===========================
Handles Firebase authentication and Firestore operations.
"""

import os
import firebase_admin
from firebase_admin import credentials, auth, firestore
from typing import Dict, Optional
from datetime import datetime
from app.config import config


# Initialize Firebase Admin SDK
_firebase_initialized = False


def initialize_firebase():
    """
    Initialize Firebase Admin SDK.
    
    This should be called once on application startup.
    """
    global _firebase_initialized
    
    if _firebase_initialized:
        return
    
    try:
        if config.FIREBASE_CREDENTIALS_PATH and os.path.exists(config.FIREBASE_CREDENTIALS_PATH):
            # Initialize with service account key file
            cred = credentials.Certificate(config.FIREBASE_CREDENTIALS_PATH)
            firebase_admin.initialize_app(cred)
            print("✓ Firebase Admin SDK initialized with credentials file")
        elif config.FIREBASE_PROJECT_ID:
            # Initialize with default credentials (for Cloud Run, etc.)
            firebase_admin.initialize_app()
            print("✓ Firebase Admin SDK initialized with default credentials")
        else:
            print("⚠ Firebase Admin SDK not initialized (credentials not provided)")
            print("  Authentication and Firestore features will be disabled")
            return
        
        _firebase_initialized = True
        
    except Exception as e:
        print(f"✗ Firebase Admin SDK initialization failed: {e}")
        print("  Authentication and Firestore features will be disabled")


def is_firebase_initialized() -> bool:
    """Check if Firebase is initialized."""
    return _firebase_initialized


async def verify_firebase_token(id_token: str) -> Dict:
    """
    Verify Firebase ID token.
    
    Args:
        id_token: Firebase ID token from client
        
    Returns:
        Dict with keys:
        - valid (bool): True if token is valid
        - uid (str): User ID (if valid)
        - email (str): User email (if valid)
        - error (str): Error message (if invalid)
    """
    if not is_firebase_initialized():
        return {
            "valid": False,
            "uid": None,
            "email": None,
            "error": "Firebase not initialized"
        }
    
    try:
        # Verify the ID token
        decoded_token = auth.verify_id_token(id_token)
        
        return {
            "valid": True,
            "uid": decoded_token.get("uid"),
            "email": decoded_token.get("email"),
            "error": None
        }
        
    except auth.InvalidIdTokenError:
        return {
            "valid": False,
            "uid": None,
            "email": None,
            "error": "Invalid ID token"
        }
    except auth.ExpiredIdTokenError:
        return {
            "valid": False,
            "uid": None,
            "email": None,
            "error": "Expired ID token"
        }
    except Exception as e:
        return {
            "valid": False,
            "uid": None,
            "email": None,
            "error": str(e)
        }


async def save_tryon_to_firestore(
    user_uid: str,
    person_image_url: str,
    garment_image_url: str,
    result_url: str,
    cloudinary_result_url: Optional[str] = None
) -> Dict:
    """
    Save try-on result to Firestore.
    
    Args:
        user_uid: Firebase user ID
        person_image_url: URL of person image
        garment_image_url: URL of garment image
        result_url: URL of result from Replicate
        cloudinary_result_url: URL of result uploaded to Cloudinary
        
    Returns:
        Dict with success status and document ID
    """
    if not is_firebase_initialized():
        return {
            "success": False,
            "error": "Firebase not initialized",
            "doc_id": None
        }
    
    try:
        db = firestore.client()
        
        # Create try-on record
        tryon_data = {
            "person_image_url": person_image_url,
            "garment_image_url": garment_image_url,
            "result_url": result_url,
            "cloudinary_url": cloudinary_result_url,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Save to users/{uid}/tryons collection
        doc_ref = db.collection("users").document(user_uid).collection("tryons").add(tryon_data)
        
        return {
            "success": True,
            "error": None,
            "doc_id": doc_ref[1].id
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "doc_id": None
        }


async def get_user_tryons(user_uid: str, limit: int = 10) -> Dict:
    """
    Get user's try-on history from Firestore.
    
    Args:
        user_uid: Firebase user ID
        limit: Maximum number of records to return
        
    Returns:
        Dict with try-on records
    """
    if not is_firebase_initialized():
        return {
            "success": False,
            "error": "Firebase not initialized",
            "tryons": []
        }
    
    try:
        db = firestore.client()
        
        # Query tryons collection
        tryons_ref = (
            db.collection("users")
            .document(user_uid)
            .collection("tryons")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        
        docs = tryons_ref.stream()
        
        tryons = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            tryons.append(data)
        
        return {
            "success": True,
            "error": None,
            "tryons": tryons
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tryons": []
        }
