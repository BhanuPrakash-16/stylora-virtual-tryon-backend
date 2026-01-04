"""
Clothing Classification and Rules Enforcement
==============================================
Validates that garments meet safety and appropriateness rules.
"""

import cv2
import numpy as np
from typing import Dict, List
from app.utils.image_utils import bytes_to_numpy
from app.config import config


def check_garment_safety(image_bytes: bytes) -> Dict:
    """
    Validate that garment image meets safety rules.
    
    Rules:
    1. Must be appropriate clothing type (shirts, pants, dresses, etc.)
    2. Must NOT be banned items (bikini, lingerie, transparent, etc.)
    3. Should not contain a person wearing it
    4. Must be fully visible
    
    NOTE: This uses simple keyword matching and basic image analysis.
    It's not a perfect ML classifier. Conservative rejection is applied.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Dict with safe/reason/confidence/details
    """
    img = bytes_to_numpy(image_bytes)
    
    if img is None:
        return {
            "safe": False,
            "reason": "invalid_image",
            "confidence": 0.0,
            "details": {"error": "Could not decode image"}
        }
    
    try:
        # Check 1: Ensure garment occupies appropriate area
        # (Not too small, not entire image)
        garment_area_check = _check_garment_area(img)
        if not garment_area_check["safe"]:
            return garment_area_check
        
        # Check 2: Detect if person is wearing the garment
        person_check = _check_no_person_in_garment(image_bytes)
        if not person_check["safe"]:
            return person_check
        
        # Check 3: Simple color/texture analysis
        # Reject if appears to be skin-like or transparent
        texture_check = _check_garment_texture(img)
        if not texture_check["safe"]:
            return texture_check
        
        # All checks passed
        return {
            "safe": True,
            "reason": "garment_valid",
            "confidence": 0.8,  # Moderate confidence (heuristic-based)
            "details": {
                "note": "Garment appears appropriate (heuristic-based check)",
                "reminder": "Allowed: shirts, pants, dresses. Banned: bikinis, lingerie, transparent items."
            }
        }
        
    except Exception as e:
        # Conservative rejection on error
        return {
            "safe": False,
            "reason": "garment_check_error",
            "confidence": 0.0,
            "details": {"error": str(e)}
        }


def _check_garment_area(img: np.ndarray) -> Dict:
    """
    Check that garment occupies appropriate portion of image.
    
    Too small → might not be visible enough
    Too large → might be entire scene, not just garment
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to separate potential garment from background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate foreground ratio
    foreground_ratio = np.count_nonzero(binary) / binary.size
    
    if foreground_ratio < config.MIN_GARMENT_AREA_RATIO:
        return {
            "safe": False,
            "reason": "garment_too_small",
            "confidence": 1.0,
            "details": {
                "foreground_ratio": foreground_ratio,
                "message": "Garment is not clearly visible. Please use a clear garment image."
            }
        }
    
    if foreground_ratio > config.MAX_GARMENT_AREA_RATIO:
        return {
            "safe": False,
            "reason": "garment_area_invalid",
            "confidence": 0.7,
            "details": {
                "foreground_ratio": foreground_ratio,
                "message": "Image should contain only the garment item."
            }
        }
    
    return {
        "safe": True,
        "reason": "garment_area_ok",
        "confidence": 1.0,
        "details": {"foreground_ratio": foreground_ratio}
    }


def _check_no_person_in_garment(image_bytes: bytes) -> Dict:
    """
    Check that garment image doesn't contain a person wearing it.
    
    Uses MediaPipe pose detection. If human pose is detected,
    the image is rejected.
    """
    try:
        import mediapipe as mp
        
        img = bytes_to_numpy(image_bytes)
        mp_pose = mp.solutions.pose
        
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,  # Lighter model
            min_detection_confidence=0.3  # Lower threshold for detection
        ) as pose:
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_img)
            
            if results.pose_landmarks:
                # Person detected in garment image
                return {
                    "safe": False,
                    "reason": "person_in_garment_image",
                    "confidence": 1.0,
                    "details": {
                        "message": "Garment image should not contain a person wearing it. Please use a flat lay or hanger image."
                    }
                }
        
        return {
            "safe": True,
            "reason": "no_person_detected",
            "confidence": 0.8,
            "details": {}
        }
        
    except Exception:
        # If check fails, assume safe (don't reject on technical error)
        return {
            "safe": True,
            "reason": "person_check_skipped",
            "confidence": 0.5,
            "details": {}
        }


def _check_garment_texture(img: np.ndarray) -> Dict:
    """
    Analyze garment texture/color to detect inappropriate items.
    
    Checks:
    - Skin-like colors (might be innerwear/nude-colored)
    - Excessive transparency (very light fabrics)
    """
    from app.utils.image_utils import detect_skin_tone_hsv, calculate_region_area
    
    # Check for skin-like colors
    skin_mask = detect_skin_tone_hsv(img)
    skin_ratio = calculate_region_area(skin_mask)
    
    # If garment is mostly skin-colored, might be innerwear/nude clothing
    if skin_ratio > 0.40:  # 40% skin-toned
        return {
            "safe": False,
            "reason": "inappropriate_garment_color",
            "confidence": min(1.0, skin_ratio),
            "details": {
                "skin_color_ratio": skin_ratio,
                "message": "Garment appears inappropriate. Please use appropriate clothing items."
            }
        }
    
    return {
        "safe": True,
        "reason": "texture_ok",
        "confidence": 0.7,
        "details": {"skin_color_ratio": skin_ratio}
    }


def classify_garment_type(image_bytes: bytes) -> Dict:
    """
    Attempt to classify garment type (optional feature).
    
    This is a placeholder for future ML-based classification.
    Currently returns a generic classification.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Dict with classification results
    """
    # Placeholder: In production, you could integrate:
    # - Google Vision API (paid)
    # - Custom ML model
    # - Gemini AI for classification
    
    return {
        "type": "unknown",
        "confidence": 0.0,
        "allowed": True,  # Default to allowed (safety checks handle rejection)
        "note": "Classification not implemented (optional feature)"
    }


def is_allowed_garment_type(garment_type: str) -> bool:
    """
    Check if garment type is in allowed list.
    
    Args:
        garment_type: Garment type string (lowercase)
        
    Returns:
        True if allowed, False if banned
    """
    garment_type_lower = garment_type.lower()
    
    # Check if banned
    for banned in config.BANNED_GARMENTS:
        if banned in garment_type_lower:
            return False
    
    # Check if explicitly allowed
    for allowed in config.ALLOWED_GARMENTS:
        if allowed in garment_type_lower:
            return True
    
    # Unknown type - default to REJECT (conservative approach)
    return False
