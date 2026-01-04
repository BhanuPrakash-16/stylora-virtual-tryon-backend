"""
Stylora Backend Configuration
==============================
Environment variables and safety thresholds.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration"""
    
    # API Configuration
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
    
    # Firebase Admin SDK
    FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "")
    FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")
    
    # Cloudinary Configuration
    CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
    CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "")
    CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Safety Configuration
    AGE_CHECK_ENABLED = os.getenv("AGE_CHECK_ENABLED", "true").lower() == "true"
    MIN_AGE = int(os.getenv("MIN_AGE", "18"))
    
    # Pose Detection Thresholds
    MIN_POSE_CONFIDENCE = float(os.getenv("MIN_POSE_CONFIDENCE", "0.5"))
    MIN_DISTANCE_M = float(os.getenv("MIN_DISTANCE_M", "1.0"))
    MAX_DISTANCE_M = float(os.getenv("MAX_DISTANCE_M", "3.0"))
    MIN_SHOULDER_WIDTH_RATIO = float(os.getenv("MIN_SHOULDER_WIDTH_RATIO", "0.15"))
    MAX_SHOULDER_WIDTH_RATIO = float(os.getenv("MAX_SHOULDER_WIDTH_RATIO", "0.60"))
    
    # Age Detection Thresholds
    MAX_HEAD_BODY_RATIO = float(os.getenv("MAX_HEAD_BODY_RATIO", "0.16"))
    
    # NSFW Detection (Skin Exposure)
    SKIN_EXPOSURE_THRESHOLD = float(os.getenv("SKIN_EXPOSURE_THRESHOLD", "0.25"))
    CONCENTRATED_SKIN_THRESHOLD = float(os.getenv("CONCENTRATED_SKIN_THRESHOLD", "0.15"))
    
    # Clothing Rules
    ALLOWED_GARMENTS = [
        "shirt", "t-shirt", "tshirt", "blouse", "top",
        "dress", "skirt",
        "pants", "jeans", "trousers", "shorts",
        "jacket", "coat", "blazer",
        "sweater", "hoodie", "cardigan",
        "suit", "formal wear"
    ]
    
    BANNED_GARMENTS = [
        "lingerie", "underwear", "bra", "panties",
        "swimwear", "bikini", "swimsuit", "bathing suit",
        "revealing", "sheer", "transparent"
    ]
    
    # Gemini AI Configuration (for safety checks)
    GEMINI_MODEL = "gemini-1.5-flash"  # Free tier model
    
    # CORS Settings
    # Allow custom origins from environment variable (comma-separated)
    # Default: localhost for development
    _default_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    _custom_origins = os.getenv("CORS_ORIGINS", "").split(",")
    _custom_origins = [origin.strip() for origin in _custom_origins if origin.strip()]
    CORS_ORIGINS = _default_origins + _custom_origins
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    
    # Image Upload Limits
    MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    
    # Safety Feedback Messages (User-Friendly)
    ERROR_MESSAGES = {
        # Age verification
        "age_uncertain": "We couldn't verify your age from this photo. Please use a clear, full-body photo.",
        "appears_minor": "This photo appears to show someone under 18. For your safety, please use a photo of an adult.",
        "no_pose_detected": "We couldn't detect your pose. Please ensure you're clearly visible in the photo.",
        "age_check_error": "Age verification failed. Please try a different photo.",
        
        # Pose and distance
        "too_close": "You appear too close to the camera. Please step back 1-2 meters.",
        "too_far": "You appear too far from the camera. Please move closer.",
        "incomplete_pose": "Full upper body not visible. Please ensure your entire torso is in frame.",
        "poor_framing": "Person appears cut off. Please ensure full body is visible in frame.",
        "pose_check_error": "Pose validation failed. Please use a clear photo with good lighting.",
        
        # NSFW content
        "excessive_skin_exposure": "This image doesn't meet our modesty standards. Please use appropriate attire.",
        "unsafe_torso_exposure": "Inappropriate clothing detected. Please wear appropriate attire.",
        "large_skin_region": "Detected large exposed skin region. Please use appropriate clothing.",
        "nsfw_check_error": "Content safety check failed. Please ensure you're wearing appropriate clothing.",
        
        # Clothing rules
        "unsafe_clothing": "This garment type isn't supported. Try shirts, dresses, pants, or jackets.",
        "transparent_clothing": "Transparent or sheer clothing is not allowed.",
        "person_in_garment_image": "The garment image should not contain a person wearing it.",
        
        # General
        "invalid_image": "Invalid image format. Please use JPEG, PNG, or WebP.",
        "general_error": "Something went wrong. Please try again with a different image.",
        "all_checks_passed": "All safety checks passed successfully."
    }
    
    # Legacy mapping (for backwards compatibility)
    SAFETY_MESSAGES = {
        "age_fail": ERROR_MESSAGES["appears_minor"],
        "nsfw_fail": ERROR_MESSAGES["excessive_skin_exposure"],
        "pose_fail": ERROR_MESSAGES["incomplete_pose"],
        "clothing_fail": ERROR_MESSAGES["unsafe_clothing"],
        "general_error": ERROR_MESSAGES["general_error"]
    }
    
    @classmethod
    def validate(cls):
        """Validate configuration on startup"""
        if not cls.REPLICATE_API_TOKEN:
            print("\n" + "="*60)
            print("⚠️  WARNING: REPLICATE_API_TOKEN not set!")
            print("="*60)
            print("Virtual try-on will not work until you add your API key.")
            print("")
            print("To get your API key:")
            print("1. Go to: https://replicate.com/account/api-tokens")
            print("2. Sign up (requires credit card)")
            print("3. Create an API token")
            print("4. Add to backend/.env file:")
            print("   REPLICATE_API_TOKEN=your_token_here")
            print("")
            print("Cost: ~$0.05 per try-on | 100 try-ons = ~$5/month")
            print("="*60 + "\n")
            # Don't raise error - allow app to start
        return True


# Export singleton instance
config = Config()
