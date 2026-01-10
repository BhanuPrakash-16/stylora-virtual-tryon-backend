"""
Stylora Virtual Try-On Backend
===============================
FastAPI application with strict safety enforcement.

ARCHITECTURE:
- All images must pass safety validation before try-on
- Free-tier only (MediaPipe, OpenCV, Replicate API)
- Conservative rejection approach (safety > convenience)
- No paid moderation APIs

SAFETY PIPELINE:
1. Image format validation
2. Person image safety (age, pose, NSFW)
3. Garment image safety (clothing rules, transparency)
4. ONLY if all pass → Replicate virtual try-on

Created: 2026-01-01
Author: Stylora Team
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import config
from app.routes import tryon


# Create FastAPI application
app = FastAPI(
    title="Stylora Virtual Try-On API",
    description="AI-powered virtual try-on with strict safety enforcement",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stylora-virtual-tryon.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(tryon.router)


# ============================================
# HEALTH CHECK & STATUS ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """
    Root endpoint - API health check.
    """
    return {
        "service": "Stylora Virtual Try-On API",
        "version": "1.0.0",
        "status": "active",
        "safety_enabled": True,
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    Ultra-light health check for Railway/Render.
    
    CRITICAL: Must be instant and never fail.
    No ML imports, no external calls, no heavy checks.
    """
    return {"status": "ok"}


@app.get("/config")
async def get_config():
    """
    Get public configuration information.
    
    Returns non-sensitive config values.
    """
    return {
        "age_check_enabled": config.AGE_CHECK_ENABLED,
        "safety_thresholds": {
            "skin_exposure_max": f"{config.SKIN_EXPOSURE_THRESHOLD * 100}%",
            "min_pose_confidence": config.MIN_POSE_CONFIDENCE,
            "shoulder_width_range": {
                "min": config.MIN_SHOULDER_WIDTH_RATIO,
                "max": config.MAX_SHOULDER_WIDTH_RATIO
            }
        },
        "allowed_garments": config.ALLOWED_GARMENTS,
        "banned_garments": config.BANNED_GARMENTS,
        "note": "These are conservative thresholds for safety-first approach"
    }


# ============================================
# STARTUP & SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """
    Run on application startup.
    
    CRITICAL: This must NEVER crash the application.
    All services are optional and wrapped in try-except.
    """
    import os
    
    print("=" * 60)
    print("Stylora Virtual Try-On Backend (ENHANCED Mode)")
    print("=" * 60)
    print(f"Version: 1.0.0 (ML-Powered)")
    print(f"Mode: Enhanced Overlay with MediaPipe & Rembg")
    print("=" * 60)
    
    # Validate configuration (never crash)
    try:
        from app.config import config
        config.validate()
        print("✓ Configuration validated")
    except Exception as e:
        print(f"⚠ Config validation skipped: {e}")
    
    # Initialize Firebase (optional)
    if os.getenv("FIREBASE_ENABLED", "false").lower() == "true":
        try:
            from app.services.firebase_service import initialize_firebase
            initialize_firebase()
            print("✓ Firebase initialized")
        except Exception as e:
            print(f"⚠ Firebase skipped: {e}")
    else:
        print("○ Firebase disabled (set FIREBASE_ENABLED=true to enable)")
    
    # Initialize Cloudinary (optional)
    if os.getenv("CLOUDINARY_ENABLED", "false").lower() == "true":
        try:
            from app.services.cloudinary_service import initialize_cloudinary
            initialize_cloudinary()
            print("✓ Cloudinary initialized")
        except Exception as e:
            print(f"⚠ Cloudinary skipped: {e}")
    else:
        print("○ Cloudinary disabled (set CLOUDINARY_ENABLED=true to enable)")
    
    print("=" * 60)
    print("API Endpoints:")
    print("  - GET  /         - Root health check")
    print("  - GET  /health   - Ultra-light health check")
    print("  - GET  /config   - Public configuration")
    print("  - POST /api/tryon - Virtual try-on (ML-Enhanced)")
    print("  - GET  /docs     - API documentation")
    print("=" * 60)
    print("✅ Backend started successfully!")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Run on application shutdown.
    """
    print("Shutting down Stylora backend...")


# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unexpected errors.
    """
    return {
        "status": "error",
        "message": "An unexpected error occurred",
        "detail": str(exc) if config.DEBUG else "Internal server error"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )
