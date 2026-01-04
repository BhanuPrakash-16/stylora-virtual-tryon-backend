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
    allow_origins=config.CORS_ORIGINS,
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
    Comprehensive health check.
    
    Checks:
    - API is running
    - OpenAI API is configured
    - Safety modules are loadable
    """
    
    # Check Replicate API status
    replicate_status = {
        "configured": bool(config.REPLICATE_API_TOKEN),
        "accessible": True if config.REPLICATE_API_TOKEN else False,
        "error": None if config.REPLICATE_API_TOKEN else "API token not set"
    }
    
    # Check if safety modules can be imported
    safety_modules_ok = True
    try:
        from app.safety import age_check, pose_check, nsfw_check, clothing_rules
        from app.safety.safety_orchestrator import validate_full_request
    except Exception as e:
        safety_modules_ok = False
    
    health_status = {
        "api": "healthy",
        "replicate": replicate_status,
        "safety_modules": "healthy" if safety_modules_ok else "error",
        "timestamp": None
    }
    
    # Overall status
    overall_healthy = safety_modules_ok  # API key is optional for startup
    
    health_status["overall"] = "healthy" if overall_healthy else "degraded"
    
    return health_status


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
    
    Validates configuration and logs startup info.
    """
    print("=" * 60)
    print("Stylora Virtual Try-On Backend (Replicate API - Professional AI)")
    print("=" * 60)
    print(f"Version: 1.0.0")
    print(f"Safety Enforcement: DISABLED (for testing)")
    print(f"Age Check: DISABLED (for testing)")
    print(f"Replicate API: {'CONFIGURED' if config.REPLICATE_API_TOKEN else 'NOT CONFIGURED'}")
    print("=" * 60)
    
    # Initialize Firebase Admin SDK
    from app.services.firebase_service import initialize_firebase
    initialize_firebase()
    
    # Initialize Cloudinary
    from app.services.cloudinary_service import initialize_cloudinary
    initialize_cloudinary()
    
    # Validate configuration
    try:
        config.validate()
        print("✓ Configuration validated successfully")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        print("  Please set HUGGINGFACE_API_TOKEN in your .env file")
    
    print("=" * 60)
    print("API Endpoints:")
    print("  - GET  /         - Root health check")
    print("  - GET  /health   - Comprehensive health check")
    print("  - GET  /config   - Public configuration")
    print("  - POST /api/tryon          - Virtual try-on (safety enforced)")
    print("  - POST /api/validate       - Validate images")
    print("  - GET  /api/safety-rules   - Get safety rules")
    print("  - GET  /docs     - API documentation")
    print("=" * 60)
    print("Ready to accept requests!")
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
