"""
Pose Detection and Distance Validation
=======================================
Validates that person is properly framed (not too close, not too far)
and has complete upper body visibility.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict
from app.utils.image_utils import bytes_to_numpy
from app.config import config


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose


def check_pose_and_distance(image_bytes: bytes) -> Dict:
    """
    Validate pose quality and camera distance.
    
    Checks:
    1. Pose Detection: Can we detect all required body landmarks?
    2. Distance: Is the person too close or too far from camera?
    3. Framing: Is full upper body visible?
    
    Distance is measured using shoulder width ratio:
    - Too close: shoulders occupy > 60% of image width
    - Too far: shoulders occupy < 15% of image width
    - Ideal: 15-60% range
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Dict with keys:
        - safe (bool): True if pose is valid and distance is good
        - reason (str): Explanation of decision
        - confidence (float): Detection confidence
        - details (dict): Additional information
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
        # Run pose detection
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=config.MIN_POSE_CONFIDENCE
        ) as pose:
            
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_img)
            
            if not results.pose_landmarks:
                return {
                    "safe": False,
                    "reason": "no_person_detected",
                    "confidence": 0.0,
                    "details": {"error": "Could not detect body pose"}
                }
            
            landmarks = results.pose_landmarks.landmark
            img_height, img_width = img.shape[:2]
            
            # Required landmarks for upper body detection
            required_landmarks = [
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP,
            ]
            
            # Check if all required landmarks are visible
            missing_landmarks = []
            for lm in required_landmarks:
                landmark = landmarks[lm.value]
                # Check visibility (MediaPipe provides visibility score)
                if landmark.visibility < config.MIN_POSE_CONFIDENCE:
                    missing_landmarks.append(lm.name)
            
            if missing_landmarks:
                return {
                    "safe": False,
                    "reason": "incomplete_pose",
                    "confidence": 0.0,
                    "details": {
                        "missing_landmarks": missing_landmarks,
                        "message": "Full upper body must be visible"
                    }
                }
            
            # Calculate shoulder width in pixels
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            shoulder_width_px = abs(
                (left_shoulder.x - right_shoulder.x) * img_width
            )
            
            # Calculate shoulder width ratio (relative to image width)
            shoulder_width_ratio = shoulder_width_px / img_width
            
            # Check if too close
            if shoulder_width_ratio > config.MAX_SHOULDER_WIDTH_RATIO:
                return {
                    "safe": False,
                    "reason": "too_close",
                    "confidence": 1.0,
                    "details": {
                        "shoulder_width_ratio": shoulder_width_ratio,
                        "threshold": config.MAX_SHOULDER_WIDTH_RATIO,
                        "message": "You appear too close to the camera. Please step back."
                    }
                }
            
            # Check if too far
            if shoulder_width_ratio < config.MIN_SHOULDER_WIDTH_RATIO:
                return {
                    "safe": False,
                    "reason": "too_far",
                    "confidence": 1.0,
                    "details": {
                        "shoulder_width_ratio": shoulder_width_ratio,
                        "threshold": config.MIN_SHOULDER_WIDTH_RATIO,
                        "message": "You appear too far from the camera. Please move closer."
                    }
                }
            
            # Check body framing (ensure person is centered and not cut off)
            # Calculate bounding box of visible landmarks
            x_coords = [lm.x for lm in landmarks if lm.visibility > 0.5]
            y_coords = [lm.y for lm in landmarks if lm.visibility > 0.5]
            
            if not x_coords or not y_coords:
                return {
                    "safe": False,
                    "reason": "invalid_framing",
                    "confidence": 0.0,
                    "details": {"error": "Could not determine body framing"}
                }
            
            # Check if person is cut off at edges
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # If person touches image edges, they might be cut off
            edge_margin = 0.05  # 5% margin
            if min_x < edge_margin or max_x > (1 - edge_margin):
                return {
                    "safe": False,
                    "reason": "poor_framing",
                    "confidence": 0.5,
                    "details": {
                        "message": "Person appears cut off. Please ensure full body is visible in frame."
                    }
                }
            
            # All checks passed
            # Calculate overall confidence as average of landmark visibilities
            avg_visibility = np.mean([
                landmarks[lm.value].visibility for lm in required_landmarks
            ])
            
            return {
                "safe": True,
                "reason": "pose_valid",
                "confidence": float(avg_visibility),
                "details": {
                    "shoulder_width_ratio": shoulder_width_ratio,
                    "avg_visibility": float(avg_visibility),
                    "landmarks_detected": len(required_landmarks)
                }
            }
            
    except Exception as e:
        return {
            "safe": False,
            "reason": "pose_check_error",
            "confidence": 0.0,
            "details": {"error": str(e)}
        }
