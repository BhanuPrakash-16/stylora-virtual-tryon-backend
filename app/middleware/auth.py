"""
Firebase Authentication Middleware
===================================
Optional authentication using Firebase JWT tokens.
"""

from fastapi import Header, HTTPException
from typing import Optional, Dict
from app.services.firebase_service import verify_firebase_token, is_firebase_initialized


async def get_current_user(
    authorization: Optional[str] = Header(None)
) -> Optional[Dict]:
    """
    Get current user from Firebase JWT token (OPTIONAL).
    
    This dependency extracts and verifies the Firebase ID token from
    the Authorization header. If no token is provided or Firebase is
    not initialized, returns None (allowing anonymous access).
    
    Args:
        authorization: Authorization header (Bearer <token>)
        
    Returns:
        Dict with user info if authenticated, None otherwise
        
    Raises:
        HTTPException: Only if token is provided but invalid
    """
    # No token provided - allow anonymous access
    if not authorization:
        return None
    
    # Firebase not initialized - allow anonymous access
    if not is_firebase_initialized():
        return None
    
    # Extract token
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format. Use: Bearer <token>"
        )
    
    token = authorization.split(" ")[1]
    
    # Verify token
    result = await verify_firebase_token(token)
    
    if not result["valid"]:
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {result['error']}"
        )
    
    return {
        "uid": result["uid"],
        "email": result["email"]
    }


async def require_auth(
    authorization: Optional[str] = Header(None)
) -> Dict:
    """
    Require authentication (STRICT).
    
    This dependency requires a valid Firebase JWT token.
    Use this for endpoints that MUST be authenticated.
    
    Args:
        authorization: Authorization header (Bearer <token>)
        
    Returns:
        Dict with user info
        
    Raises:
        HTTPException: If not authenticated or token invalid
    """
    user = await get_current_user(authorization)
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please provide a valid Firebase token."
        )
    
    return user
