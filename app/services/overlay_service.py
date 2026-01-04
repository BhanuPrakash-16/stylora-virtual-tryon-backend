"""
Stylora: Geometry-Based Virtual Try-On Service (Neck & Coverage Fix)
====================================================================
1. Ghost Mannequin: Conservative neck removal to preserve collars/blouses.
2. Geometry Magic: Aligns strictly to shoulders for full T-shirt coverage.
3. Full Overlay: No holes for hands; garment fully covers the user.
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter
import mediapipe as mp
import base64
import io
from rembg import remove
from typing import Dict, Tuple, Optional, List

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# ==========================================
# 1. CORE MATH & DETECTION
# ==========================================

def get_landmarks(image_cv: np.ndarray) -> Optional[Dict]:
    """
    Detects landmarks and returns them as a dictionary of coordinate tuples.
    """
    h, w = image_cv.shape[:2]
    img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
        results = pose.process(img_rgb)
        
        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark
        
        # Helper to get coords
        def loc(landmark): return (int(landmark.x * w), int(landmark.y * h))
        
        # Extract raw points
        l_sh = loc(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
        r_sh = loc(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
        l_hip = loc(lm[mp_pose.PoseLandmark.LEFT_HIP])
        r_hip = loc(lm[mp_pose.PoseLandmark.RIGHT_HIP])
        
        return {
            "nose": loc(lm[mp_pose.PoseLandmark.NOSE]),
            "l_sh": l_sh,
            "r_sh": r_sh,
            "l_elbow": loc(lm[mp_pose.PoseLandmark.LEFT_ELBOW]),
            "r_elbow": loc(lm[mp_pose.PoseLandmark.RIGHT_ELBOW]),
            "l_wrist": loc(lm[mp_pose.PoseLandmark.LEFT_WRIST]),
            "r_wrist": loc(lm[mp_pose.PoseLandmark.RIGHT_WRIST]),
            "l_hip": l_hip,
            "r_hip": r_hip,
            # Calculated Center Points
            "shoulder_center": (
                int((l_sh[0] + r_sh[0]) * 0.5),
                int((l_sh[1] + r_sh[1]) * 0.5)
            ),
            "hip_center": (
                int((l_hip[0] + r_hip[0]) * 0.5),
                int((l_hip[1] + r_hip[1]) * 0.5)
            ),
            # Add shoulder width for scaling fallback
            "shoulder_width": np.linalg.norm(np.array(l_sh) - np.array(r_sh))
        }

# ==========================================
# 2. GHOST MANNEQUIN ENGINE (NECK FIX)
# ==========================================

def create_ghost_garment(garment_bytes: bytes) -> Tuple[Image.Image, Optional[Dict]]:
    """
    Removes ONLY the model's head.
    CRITICAL FIX: The neck erasure now starts HIGHER up to preserve collars/blouses.
    """
    # 1. Decode Image for OpenCV
    nparr = np.frombuffer(garment_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Detect Landmarks (Do this BEFORE removing background)
    landmarks = get_landmarks(img_cv)
    
    # 3. Remove Background (Standard Rembg)
    print("🎨 Removing background from garment...")
    no_bg_bytes = remove(garment_bytes)
    img_pil = Image.open(io.BytesIO(no_bg_bytes)).convert("RGBA")
    img_np = np.array(img_pil)  # Convert to numpy for editing
    
    if landmarks:
        print("✂️  Model detected! Performing conservative Ghost Mannequin extraction...")
        
        # --- ERASE HEAD & NECK (CONSERVATIVE MODE) ---
        nose = landmarks['nose']
        l_sh = landmarks['l_sh']
        r_sh = landmarks['r_sh']
        
        # FIX: Move the base of the neck triangle UP.
        # Instead of starting at the shoulders (which cuts the collar), 
        # we start 30% of the way up towards the nose.
        
        # Calculate a point "above" the shoulders for the triangle base
        shoulder_mid = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2)
        
        # Move base up by 25% of distance to nose
        # This preserves the garment sitting on the shoulders/traps
        neck_base_l = (
            int(l_sh[0] + (nose[0] - l_sh[0]) * 0.25),
            int(l_sh[1] + (nose[1] - l_sh[1]) * 0.25)
        )
        neck_base_r = (
            int(r_sh[0] + (nose[0] - r_sh[0]) * 0.25),
            int(r_sh[1] + (nose[1] - r_sh[1]) * 0.25)
        )
        
        # Draw the triangle to erase skin
        neck_triangle = np.array([neck_base_l, neck_base_r, nose], np.int32)
        cv2.drawContours(img_np, [neck_triangle], 0, (0, 0, 0, 0), -1) 
        
        # Erase Head (Circle)
        # Radius is distance from nose to new neck base
        neck_mid_raised = ((neck_base_l[0] + neck_base_r[0])//2, (neck_base_l[1] + neck_base_r[1])//2)
        head_radius = int(np.linalg.norm(np.array(nose) - np.array(neck_mid_raised)) * 1.5)
        cv2.circle(img_np, nose, head_radius, (0, 0, 0, 0), -1)
        
        # --- NO HAND ERASURE ---
        # We deliberately DO NOT erase hands so the garment stays solid.
        
        # --- CROP TO CONTENT ---
        alpha_channel = img_np[:, :, 3]
        coords = cv2.findNonZero(alpha_channel)
        
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Generous padding to keep everything
            pad = 50
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = w + 2 * pad
            h = h + 2 * pad
            
            final_img = Image.fromarray(img_np).crop((x, y, x+w, y+h))
            
            # Adjust landmarks
            adjusted_landmarks = {}
            for k, v in landmarks.items():
                if isinstance(v, tuple):
                    adjusted_landmarks[k] = (v[0] - x, v[1] - y)
                else:
                    adjusted_landmarks[k] = v
                
            return final_img, adjusted_landmarks
        
        return img_pil, landmarks

    else:
        print("⚠️ No model detected in garment. Using simple background removal.")
        return img_pil, None

# ==========================================
# 3. GEOMETRY MAGIC (WARPING FIX)
# ==========================================

def warp_garment_to_user(garment_pil: Image.Image, src_lm: Dict, user_lm: Dict, target_size: Tuple[int, int]) -> Image.Image:
    """
    Applies Affine Transformation.
    FIX: Removed 'neck drop' to ensure garment sits high enough to cover t-shirt.
    """
    w, h = target_size
    garment_np = np.array(garment_pil)
    
    # --- 1. Source Points (The Garment/Model) ---
    src_l = np.array(src_lm['l_sh'])
    src_r = np.array(src_lm['r_sh'])
    src_hip = np.array(src_lm['hip_center'])
    
    src_tri = np.float32([src_l, src_r, src_hip])
    
    # --- 2. Destination Points (The User) ---
    dst_l_raw = np.array(user_lm['l_sh'])
    dst_r_raw = np.array(user_lm['r_sh'])
    dst_hip_raw = np.array(user_lm['hip_center'])
    
    # --- FIT CORRECTION 1: SHOULDER WIDENING ---
    # Slight widening (1.1x) to ensure sleeves cover the person's arms
    shoulder_vec = dst_r_raw - dst_l_raw
    shoulder_center = (dst_l_raw + dst_r_raw) / 2.0
    
    scale_factor = 1.15
    dst_l = shoulder_center - (shoulder_vec * scale_factor / 2.0)
    dst_r = shoulder_center + (shoulder_vec * scale_factor / 2.0)
    
    # --- FIT CORRECTION 2: POSITIONING (THE FIX) ---
    # PREVIOUSLY: We added 'neck_drop_px' which pushed the shirt down.
    # FIX: We REMOVE that drop. We align exactly to the shoulder line or slightly HIGHER.
    
    # To be safe and ensure T-shirt coverage, we push the garment UP very slightly (negative drop)
    torso_vec = dst_hip_raw - shoulder_center
    torso_len = np.linalg.norm(torso_vec)
    
    # Lift the garment up by 2% of torso length to ensure collar coverage
    lift_px = torso_len * 0.02
    dst_l[1] -= lift_px
    dst_r[1] -= lift_px
    
    # --- FIT CORRECTION 3: HEM LENGTH ---
    # Extend bottom to cover waist/belt
    dst_hip = dst_hip_raw + (torso_vec / torso_len) * (torso_len * 0.15)
    
    dst_tri = np.float32([dst_l, dst_r, dst_hip])
    
    # --- 3. Compute & Apply Warp ---
    matrix = cv2.getAffineTransform(src_tri, dst_tri)
    
    warped_np = cv2.warpAffine(
        garment_np, 
        matrix, 
        (w, h), 
        flags=cv2.INTER_LANCZOS4, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0, 0)
    )
    
    return Image.fromarray(warped_np)

# ==========================================
# 4. MAIN PIPELINE (Entry Point)
# ==========================================

async def create_simple_overlay(person_bytes: bytes, garment_bytes: bytes) -> Dict:
    """
    The main function called by the API.
    """
    try:
        print("\n=== STARTING GEOMETRY VTON (Coverage Fix) ===")
        
        # 1. Load User Image
        nparr = np.frombuffer(person_bytes, np.uint8)
        person_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if person_cv is None:
            return {"success": False, "error": "Invalid person image"}
            
        person_h, person_w = person_cv.shape[:2]
        person_pil = Image.fromarray(cv2.cvtColor(person_cv, cv2.COLOR_BGR2RGBA))
        
        # 2. Analyze User Pose
        user_landmarks = get_landmarks(person_cv)
        if not user_landmarks:
            print("⚠️ Could not detect person pose.")
            return {"success": False, "error": "Could not detect a person in the user image."}
        
        # 3. Process Garment (Ghost Mannequin)
        clean_garment, garment_landmarks = create_ghost_garment(garment_bytes)
        
        # 4. Geometry Magic
        if garment_landmarks:
            print("✨ Applying Affine Warping (High Position)...")
            final_garment = warp_garment_to_user(
                clean_garment, 
                garment_landmarks, 
                user_landmarks, 
                (person_w, person_h)
            )
        else:
            print("⚠️ Fallback: Simple Scaling...")
            target_width = int(user_landmarks['shoulder_width'] * 1.6)
            ratio = clean_garment.height / clean_garment.width
            target_height = int(target_width * ratio)
            
            clean_garment = clean_garment.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            final_garment = Image.new('RGBA', (person_w, person_h), (0,0,0,0))
            pos_x = user_landmarks['shoulder_center'][0] - target_width // 2
            # Sit higher for fallback too
            pos_y = user_landmarks['shoulder_center'][1] - int(target_height * 0.20)
            final_garment.paste(clean_garment, (pos_x, pos_y))

        # 5. Composite (Overlay)
        # Simply overlay garment ON TOP of person. No masking of person's hands.
        result = Image.alpha_composite(person_pil, final_garment)
        
        # 6. Output
        buf = io.BytesIO()
        result.convert("RGB").save(buf, format='PNG')
        result_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        print("=== SUCCESS ===\n")
        return {
            "success": True,
            "result_url": f"data:image/png;base64,{result_b64}",
            "result_base64": result_b64,
            "method": "Geometry Warp (Full Coverage)"
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}