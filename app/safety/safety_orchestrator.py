"""
Safety Orchestrator
===================
Coordinates all safety checks in proper sequence.
Implements fail-fast approach: stops at first failure.
"""

from typing import Dict
from app.safety.age_check import check_age
from app.safety.pose_check import check_pose_and_distance
from app.safety.nsfw_check import check_nsfw_content, check_transparency
from app.safety.clothing_rules import check_garment_safety
from app.config import config


async def validate_person_image(image_bytes: bytes) -> Dict:
    """
    Run ALL person image safety checks in sequence.
    
    Pipeline:
    1. Age verification (18+ only)
    2. Pose and distance validation
    3. NSFW content check
    
    Fail-fast: Stops at first check that fails.
    
    Args:
        image_bytes: Raw person image bytes
        
    Returns:
        Dict with keys:
        - safe (bool): True if all checks pass
        - stage (str): Which check failed (if any)
        - reason (str): Machine-readable reason code
        - message (str): User-friendly message
        - details (dict): Additional information
    """
    
    # Stage 1: Age Check
    age_result = check_age(image_bytes)
    if not age_result["safe"]:
        return {
            "safe": False,
            "stage": "age_verification",
            "reason": age_result["reason"],
            "message": _get_user_message(age_result["reason"]),
            "confidence": age_result["confidence"],
            "details": age_result["details"]
        }
    
    # Stage 2: Pose & Distance Check
    pose_result = check_pose_and_distance(image_bytes)
    if not pose_result["safe"]:
        return {
            "safe": False,
            "stage": "pose_validation",
            "reason": pose_result["reason"],
            "message": _get_user_message(pose_result["reason"]),
            "confidence": pose_result["confidence"],
            "details": pose_result["details"]
        }
    
    # Stage 3: NSFW Content Check
    nsfw_result = check_nsfw_content(image_bytes)
    if not nsfw_result["safe"]:
        return {
            "safe": False,
            "stage": "content_safety",
            "reason": nsfw_result["reason"],
            "message": _get_user_message(nsfw_result["reason"]),
            "confidence": nsfw_result["confidence"],
            "details": nsfw_result["details"]
        }
    
    # All checks passed
    return {
        "safe": True,
        "stage": "complete",
        "reason": "all_checks_passed",
        "message": "Person image is valid and safe.",
        "confidence": min(
            age_result["confidence"],
            pose_result["confidence"],
            nsfw_result["confidence"]
        ),
        "details": {
            "age_check": age_result,
            "pose_check": pose_result,
            "nsfw_check": nsfw_result
        }
    }


async def validate_garment_image(image_bytes: bytes) -> Dict:
    """
    Run ALL garment image safety checks.
    
    Pipeline:
    1. Garment safety rules (no bikinis, lingerie, etc.)
    2. Transparency check
    3. No person in garment image
    
    Args:
        image_bytes: Raw garment image bytes
        
    Returns:
        Dict with safe/stage/reason/message/details
    """
    
    # Stage 1: Garment Safety Rules
    garment_result = check_garment_safety(image_bytes)
    if not garment_result["safe"]:
        return {
            "safe": False,
            "stage": "garment_validation",
            "reason": garment_result["reason"],
            "message": _get_user_message(garment_result["reason"]),
            "confidence": garment_result["confidence"],
            "details": garment_result["details"]
        }
    
    # Stage 2: Transparency Check
    transparency_result = check_transparency(image_bytes)
    if not transparency_result["safe"]:
        return {
            "safe": False,
            "stage": "transparency_check",
            "reason": transparency_result["reason"],
            "message": _get_user_message(transparency_result["reason"]),
            "confidence": transparency_result["confidence"],
            "details": transparency_result["details"]
        }
    
    # All checks passed
    return {
        "safe": True,
        "stage": "complete",
        "reason": "all_checks_passed",
        "message": "Garment image is valid and safe.",
        "confidence": min(
            garment_result["confidence"],
            transparency_result["confidence"]
        ),
        "details": {
            "garment_check": garment_result,
            "transparency_check": transparency_result
        }
    }


async def validate_full_request(person_bytes: bytes, garment_bytes: bytes) -> Dict:
    """
    Validate both person and garment images together.
    
    This is the main entry point for full validation before try-on.
    
    Args:
        person_bytes: Raw person image bytes
        garment_bytes: Raw garment image bytes
        
    Returns:
        Dict with safe/reason/message/details
    """
    
    # Validate person image
    person_result = await validate_person_image(person_bytes)
    if not person_result["safe"]:
        return {
            "safe": False,
            "failed_image": "person",
            **person_result
        }
    
    # Validate garment image
    garment_result = await validate_garment_image(garment_bytes)
    if not garment_result["safe"]:
        return {
            "safe": False,
            "failed_image": "garment",
            **garment_result
        }
    
    # Both passed
    return {
        "safe": True,
        "failed_image": None,
        "stage": "complete",
        "reason": "all_validations_passed",
        "message": "All safety checks passed. Ready for virtual try-on.",
        "details": {
            "person_validation": person_result,
            "garment_validation": garment_result
        }
    }


def _get_user_message(reason_code: str) -> str:
    """
    Convert machine-readable reason code to user-friendly message.
    
    Args:
        reason_code: Reason code from safety check
        
    Returns:
        User-friendly error message
    """
    return config.ERROR_MESSAGES.get(
        reason_code,
        "Safety validation failed. Please ensure you're using appropriate images."
    )
