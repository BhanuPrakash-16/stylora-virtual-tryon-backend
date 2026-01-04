"""
Age Verification Module
========================
Uses conservative heuristics to verify age (18+).

DISCLAIMER: Age detection is NOT perfect. This uses body proportion
heuristics which can have false positives/negatives. We use conservative
thresholds to err on the side of caution.

When age is uncertain → REJECT (safety-first approach)
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict
from app.utils.image_utils import bytes_to_numpy
from app.config import config


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose


def check_age(image_bytes: bytes) -> Dict:
    """
    Verify that the person in the image appears to be an adult (18+).
    
    Method:
    - Extract body landmarks using MediaPipe Pose
    - Calculate head-to-body ratio
    - Children typically have larger head-to-body ratios (>0.15)
    - Adults typically have smaller ratios (<0.16)
    
    IMPORTANT: This is a heuristic approach with limitations.
    - Can fail with unusual poses or camera angles
    - May incorrectly classify petite adults or tall teens
    - Should be combined with other safety checks
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Dict with keys:
        - safe (bool): True if appears to be adult, False otherwise
        - reason (str): Explanation of decision
        - confidence (float): Confidence score (0-1)
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
    
    # Disable age check if configured
    if not config.AGE_CHECK_ENABLED:
        return {
            "safe": True,
            "reason": "age_check_disabled",
            "confidence": 1.0,
            "details": {}
        }
    
    try:
        # Run pose detection
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        ) as pose:
            
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_img)
            
            if not results.pose_landmarks:
                # No pose detected - conservative rejection
                return {
                    "safe": False,
                    "reason": "no_pose_detected",
                    "confidence": 0.0,
                    "details": {"error": "Could not detect body landmarks"}
                }
            
            landmarks = results.pose_landmarks.landmark
            img_height, img_width = img.shape[:2]
            
            # Extract key landmarks
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Calculate head position (top of head approximation)
            # Use nose + offset as approximation of head top
            head_top_y = nose.y - 0.05  # Approximate head top
            
            # Calculate shoulder midpoint
            shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # Calculate hip midpoint
            hip_mid_y = (left_hip.y + right_hip.y) / 2
            
            # Calculate proportions
            head_to_shoulder = abs(shoulder_mid_y - head_top_y)
            shoulder_to_hip = abs(hip_mid_y - shoulder_mid_y)
            total_body_height = head_to_shoulder + shoulder_to_hip
            
            if total_body_height == 0:
                return {
                    "safe": False,
                    "reason": "invalid_proportions",
                    "confidence": 0.0,
                    "details": {"error": "Could not calculate body proportions"}
                }
            
            # Head-to-body ratio
            head_body_ratio = head_to_shoulder / total_body_height
            
            # Conservative thresholds:
            # - Children typically have ratio > 0.18
            # - Adults typically have ratio < 0.16
            # - If ratio is in uncertain range (0.16-0.18), we REJECT
            
            is_adult = head_body_ratio < config.MAX_HEAD_BODY_RATIO
            is_child = head_body_ratio > config.MAX_HEAD_BODY_RATIO
            
            # Calculate confidence
            # The further from the threshold, the more confident we are
            if is_adult:
                confidence = min(1.0, (config.MAX_HEAD_BODY_RATIO - head_body_ratio) / 0.06)
            else:
                confidence = min(1.0, (head_body_ratio - config.MAX_HEAD_BODY_RATIO) / 0.06)
            
            # If confidence is low (uncertain), we REJECT
            if confidence < 0.5:
                return {
                    "safe": False,
                    "reason": "age_uncertain",
                    "confidence": confidence,
                    "details": {
                        "head_body_ratio": head_body_ratio,
                        "threshold": config.MAX_HEAD_BODY_RATIO,
                        "note": "Uncertain age - rejecting for safety"
                    }
                }
            
            if is_child:
                return {
                    "safe": False,
                    "reason": "appears_minor",
                    "confidence": confidence,
                    "details": {
                        "head_body_ratio": head_body_ratio,
                        "threshold": config.MAX_HEAD_BODY_RATIO
                    }
                }
            
            return {
                "safe": True,
                "reason": "appears_adult",
                "confidence": confidence,
                "details": {
                    "head_body_ratio": head_body_ratio,
                    "threshold": config.MAX_HEAD_BODY_RATIO
                }
            }
            
    except Exception as e:
        # Any error in age detection → conservative rejection
        return {
            "safe": False,
            "reason": "age_check_error",
            "confidence": 0.0,
            "details": {"error": str(e)}
        }
