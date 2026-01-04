# Stylora Backend API

A **FastAPI-powered backend** for AI-driven virtual try-on with comprehensive safety checks. This backend provides a professional, production-ready API that applies strict content moderation before processing try-on requests.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Architecture](#-architecture)
- [Getting Started](#-getting-started)
- [API Endpoints](#-api-endpoints)
- [Safety System](#-safety-system)
- [Configuration](#-configuration)
- [Optional Integrations](#-optional-integrations)
- [Development](#-development)
- [Deployment](#-deployment)

---

## 🎯 Overview

Stylora's backend is designed to provide **safe, reliable virtual try-on experiences** by combining:
- **Geometry-based overlay system** for instant, free virtual try-on
- **Multi-layer safety validation** using MediaPipe and OpenCV
- **Optional cloud integrations** (Firebase, Cloudinary, Replicate)
- **Production-grade error handling** and logging

### Purpose

This API enables users to visualize how garments look on them by:
1. Accepting person and garment images
2. Running comprehensive safety checks (age, pose, content)
3. Creating realistic overlays using pose detection and geometry
4. Returning the try-on result (optionally stored in cloud)

---

## ✨ Features

### Core Capabilities
- ✅ **Virtual Try-On**: Geometry-based garment overlay using MediaPipe pose detection
- ✅ **Safety-First**: Multi-stage validation (age, pose, NSFW, clothing rules)
- ✅ **Free-Tier Friendly**: Runs entirely on local processing (no paid APIs required)
- ✅ **Fast Processing**: Instant overlay rendering (~1-2 seconds)
- ✅ **RESTful API**: Clean, documented endpoints with FastAPI

### Optional Features
- 📸 **Cloudinary Integration**: Upload and host result images
- 🔥 **Firebase Integration**: User authentication and try-on history storage
- 🎨 **Replicate API Support**: Professional AI-powered try-on (when configured)

---

## 🛠 Technology Stack

### Core Technologies
| Technology | Version | Purpose |
|-----------|---------|---------|
| **FastAPI** | 0.109.0 | Web framework and API server |
| **Uvicorn** | 0.27.0 | ASGI server for production |
| **OpenCV** | 4.9.0 | Image processing and transformations |
| **MediaPipe** | 0.10.9 | Pose detection and landmark extraction |
| **NumPy** | 1.24.3 | Numerical computations |
| **Pillow** | 10.2.0 | Image manipulation |
| **Rembg** | 2.0.56 | Background removal for garment extraction |

### Optional Integrations
| Service | Purpose |
|---------|---------|
| **Replicate** | Professional AI try-on (paid, optional) |
| **Firebase Admin SDK** | Authentication and Firestore database |
| **Cloudinary** | Cloud image hosting and CDN |
| **Google Gemini AI** | Advanced safety checks (optional) |

### Python Core
- **Python 3.8+** required
- **python-dotenv** for environment configuration
- **python-multipart** for file upload handling

---

## 🏗 Architecture

### Request Flow

```
┌─────────────┐
│   Client    │
│  (Web/App)  │
└──────┬──────┘
       │ POST /api/tryon
       │ (person + garment images)
       ▼
┌─────────────────┐
│  FastAPI Server │
└────────┬────────┘
         │
    ┌────▼─────────────────────────────────┐
    │  Safety Validation Pipeline          │
    │  ----------------------------------- │
    │  1. Image Format Validation          │
    │  2. Person Image Safety Check        │
    │     - Age Verification (18+)         │
    │     - Pose & Distance Check          │
    │     - NSFW Content Detection         │
    │  3. Garment Image Safety Check       │
    │     - Allowed Clothing Types         │
    │     - Transparency Detection         │
    │     - No Person in Garment           │
    └────────┬─────────────────────────────┘
             │
        ┌────▼────┐
        │  Safe?  │
        └─┬─────┬─┘
     NO   │     │   YES
    ┌─────▼──┐  │
    │ REJECT │  │
    │ Return │  │
    │ Error  │  │
    └────────┘  │
                ▼
         ┌──────────────┐
         │ Overlay      │
         │ Processing   │
         │ ------------ │
         │ - MediaPipe  │
         │   Landmarks  │
         │ - Garment    │
         │   Warping    │
         │ - Composite  │
         └──────┬───────┘
                │
         ┌──────▼──────────┐
         │  Cloud Upload   │
         │  (Optional)     │
         │  -------------- │
         │  - Cloudinary   │
         │  - Firestore    │
         └──────┬──────────┘
                │
         ┌──────▼──────┐
         │   Return    │
         │   Result    │
         │   (base64)  │
         └─────────────┘
```

### Project Structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI application entry point
│   ├── config.py                # Configuration and environment variables
│   │
│   ├── routes/
│   │   └── tryon.py             # Virtual try-on endpoints
│   │
│   ├── safety/                  # Safety validation modules
│   │   ├── safety_orchestrator.py    # Coordinates all safety checks
│   │   ├── age_check.py              # Age verification (18+)
│   │   ├── pose_check.py             # Pose and distance validation
│   │   ├── nsfw_check.py             # NSFW content detection
│   │   └── clothing_rules.py         # Garment type validation
│   │
│   ├── services/                # Core services
│   │   ├── overlay_service.py        # Geometry-based try-on engine
│   │   ├── image_service.py          # Image validation and processing
│   │   ├── firebase_service.py       # Firebase integration (optional)
│   │   ├── cloudinary_service.py     # Cloudinary uploads (optional)
│   │   └── replicate_service.py      # Replicate API (optional)
│   │
│   ├── middleware/              # Authentication middleware
│   │   └── auth.py
│   │
│   └── utils/                   # Helper utilities
│       └── image_utils.py
│
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8 or higher**
- **pip** package manager
- **Virtual environment** (recommended)

### Installation

#### 1. Clone the Repository (if not already done)
```bash
cd Stylora/backend
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure Environment Variables
```bash
# Copy the example environment file
copy .env.example .env

# Edit .env and configure required variables
# - REPLICATE_API_TOKEN (optional, for AI-powered try-on)
# - Other optional services (Firebase, Cloudinary)
```

#### 5. Run the Development Server
```bash
# Option 1: Using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Option 2: Using Python
python -m app.main

# Option 3: Using the module directly
python app/main.py
```

#### 6. Verify Installation
Open your browser and navigate to:
- **API Health**: http://localhost:8000/
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

---

## 📡 API Endpoints

### Base URL
```
http://localhost:8000
```

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root health check |
| `GET` | `/health` | Comprehensive health status |
| `GET` | `/config` | Public configuration info |
| `POST` | `/api/tryon` | **Virtual try-on** (main endpoint) |
| `POST` | `/api/validate` | Pre-validate images |
| `GET` | `/api/safety-rules` | Get safety rules and policies |
| `GET` | `/docs` | Interactive API documentation |

---

### 1. Virtual Try-On

**Endpoint:** `POST /api/tryon`

Upload person and garment images to get a virtual try-on result.

#### Request

**Content-Type:** `multipart/form-data`

**Parameters:**
- `person_image` (file, required): Full-body photo of a person
- `garment_image` (file, required): Clothing item image

**Example using cURL:**
```bash
curl -X POST http://localhost:8000/api/tryon \
  -F "person_image=@person.jpg" \
  -F "garment_image=@shirt.png"
```

**Example using Python:**
```python
import requests

url = "http://localhost:8000/api/tryon"

files = {
    'person_image': open('person.jpg', 'rb'),
    'garment_image': open('shirt.png', 'rb')
}

response = requests.post(url, files=files)
result = response.json()

print(result)
```

#### Response (Success)
```json
{
  "status": "success",
  "result_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "result_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "message": "Virtual try-on completed successfully",
  "safety_checks_passed": true,
  "cloudinary_url": "https://res.cloudinary.com/...",
  "saved_to_history": true,
  "firestore_doc_id": "abc123..."
}
```

#### Response (Rejected - Safety Check Failed)
```json
{
  "status": "rejected",
  "reason": "appears_minor",
  "message": "This photo appears to show someone under 18. For your safety, please use a photo of an adult.",
  "stage": "age_verification",
  "confidence": 0.87,
  "details": {
    "head_body_ratio": 0.19,
    "threshold": 0.16
  }
}
```

#### Rejection Reasons

| Reason Code | Description |
|-------------|-------------|
| `invalid_image` | Image format not supported |
| `appears_minor` | Person appears under 18 |
| `age_uncertain` | Cannot verify age (conservative rejection) |
| `no_pose_detected` | Cannot detect body landmarks |
| `too_close` | Person too close to camera |
| `too_far` | Person too far from camera |
| `incomplete_pose` | Full upper body not visible |
| `excessive_skin_exposure` | Inappropriate clothing on person |
| `unsafe_clothing` | Banned garment type (lingerie, swimwear, etc.) |
| `transparent_clothing` | Sheer/transparent garment |

---

### 2. Validate Images

**Endpoint:** `POST /api/validate`

Pre-validate images without running the full try-on process.

#### Request
```bash
curl -X POST http://localhost:8000/api/validate \
  -F "person_image=@person.jpg" \
  -F "garment_image=@shirt.png"
```

#### Response
```json
{
  "person_image": {
    "safe": true,
    "stage": "complete",
    "confidence": 0.92,
    "details": { ... }
  },
  "garment_image": {
    "safe": false,
    "stage": "garment_validation",
    "reason": "unsafe_clothing",
    "message": "This garment type is not allowed. Only shirts, pants, dresses permitted."
  }
}
```

---

### 3. Get Safety Rules

**Endpoint:** `GET /api/safety-rules`

Retrieve information about safety policies and content rules.

#### Response
```json
{
  "age_restriction": "18+ only",
  "allowed_garments": [
    "shirt", "t-shirt", "blouse", "dress", 
    "pants", "jeans", "skirt", "jacket", 
    "coat", "sweater", "hoodie"
  ],
  "banned_garments": [
    "lingerie", "underwear", "swimwear", 
    "bikini", "swimsuit"
  ],
  "distance_rules": {
    "description": "Person should be clearly visible, not too close or too far",
    "min_shoulder_width_ratio": 0.15,
    "max_shoulder_width_ratio": 0.60
  },
  "content_policy": {
    "description": "Only appropriate, non-revealing clothing is allowed",
    "max_skin_exposure": "25%",
    "transparency": "Not allowed"
  }
}
```

---

### 4. Health Check

**Endpoint:** `GET /health`

Comprehensive health status of all services.

#### Response
```json
{
  "api": "healthy",
  "replicate": {
    "configured": true,
    "accessible": true,
    "error": null
  },
  "safety_modules": "healthy",
  "overall": "healthy"
}
```

---

## 🛡 Safety System

Stylora implements a **multi-layer safety validation pipeline** to ensure appropriate content.

### Safety Philosophy

> **Conservative Rejection**: When uncertain, we reject. Safety > convenience.

### Validation Stages

#### 1. **Age Verification** (`age_check.py`)

**Method**: Body proportion analysis using MediaPipe
- Calculates head-to-body ratio
- Adults: ratio < 0.16
- Children: ratio > 0.16
- Uncertain: REJECT

**Limitations**:
- Not 100% accurate (heuristic-based)
- Can fail with unusual poses or angles
- Designed to be conservative

#### 2. **Pose & Distance Check** (`pose_check.py`)

**Validates**:
- All required landmarks are visible
- Shoulder width ratio (distance estimation)
  - Too close: > 60% of image width → REJECT
  - Too far: < 15% of image width → REJECT
  - Ideal: 15-60% range
- Full upper body visibility
- Proper framing (not cut off)

**Technologies**: MediaPipe Pose Detection

#### 3. **NSFW Content Detection** (`nsfw_check.py`)

**Method**: Skin exposure heuristics using HSV color space
- Detects skin-tone regions
- Calculates total skin exposure ratio
- Analyzes skin distribution in torso
- Checks for large contiguous skin regions

**Thresholds**:
- Total skin exposure > 25% → REJECT
- Concentrated torso skin > 15% → REJECT
- Single skin region > 20% → REJECT

**Limitations**:
- Color-based (not ML)
- Can flag faces/arms as unsafe
- Conservative to prioritize safety

#### 4. **Clothing Rules** (`clothing_rules.py`)

**Allowed Garments**:
- Shirts, t-shirts, blouses
- Dresses, skirts
- Pants, jeans, trousers
- Jackets, coats
- Sweaters, hoodies

**Banned Garments**:
- Lingerie
- Underwear
- Swimwear (bikinis, swimsuits)
- Any sheer/transparent clothing

#### 5. **Transparency Check**

**Detects**:
- Alpha channel (RGBA images)
- Average opacity < 80% → REJECT

### Safety Configuration

All thresholds are configurable in `config.py`:

```python
# Age Check
AGE_CHECK_ENABLED = True
MIN_AGE = 18
MAX_HEAD_BODY_RATIO = 0.16

# Pose Detection
MIN_POSE_CONFIDENCE = 0.5
MIN_SHOULDER_WIDTH_RATIO = 0.15
MAX_SHOULDER_WIDTH_RATIO = 0.60

# NSFW Detection
SKIN_EXPOSURE_THRESHOLD = 0.25
CONCENTRATED_SKIN_THRESHOLD = 0.15
```

---

## ⚙ Configuration

### Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# ==========================================
# Stylora Backend Configuration
# ==========================================

# API Tokens (Optional)
REPLICATE_API_TOKEN=r8_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Firebase (Optional)
FIREBASE_CREDENTIALS_PATH=/path/to/serviceAccountKey.json
FIREBASE_PROJECT_ID=your-project-id

# Cloudinary (Optional)
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=123456789012345
CLOUDINARY_API_SECRET=xxxxxxxxxxxxxxxxxxxx

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Safety Configuration
AGE_CHECK_ENABLED=true
MIN_POSE_CONFIDENCE=0.5
SKIN_EXPOSURE_THRESHOLD=0.25

# CORS (Frontend URLs)
# Automatically includes:
# - http://localhost:3000
# - http://localhost:5173
```

### Configuration Options

#### Server Settings
- `HOST`: Server bind address (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `DEBUG`: Enable debug mode (default: `false`)

#### Safety Settings
- `AGE_CHECK_ENABLED`: Enable/disable age verification (default: `true`)
- `MIN_AGE`: Minimum age requirement (default: `18`)
- `MIN_POSE_CONFIDENCE`: Pose detection confidence threshold (default: `0.5`)
- `SKIN_EXPOSURE_THRESHOLD`: Max skin exposure ratio (default: `0.25`)

#### Rate Limiting (Optional)
- `RATE_LIMIT_ENABLED`: Enable rate limiting (default: `false`)
- `RATE_LIMIT_PER_MINUTE`: Requests per minute (default: `10`)

---

## 🔌 Optional Integrations

### 1. Replicate API (AI-Powered Try-On)

**Purpose**: Professional AI-powered virtual try-on using Kolors model

**Setup**:
1. Create account at https://replicate.com
2. Get API token from https://replicate.com/account/api-tokens
3. Add to `.env`: `REPLICATE_API_TOKEN=r8_xxx...`

**Cost**: ~$0.05 per try-on | 100 try-ons = ~$5/month

**Note**: Not required for basic overlay-based try-on

---

### 2. Firebase (Authentication & Database)

**Purpose**: User authentication and try-on history storage

**Setup**:
1. Create Firebase project at https://console.firebase.google.com
2. Download service account key (Settings → Service Accounts)
3. Add to `.env`:
   ```env
   FIREBASE_CREDENTIALS_PATH=/path/to/serviceAccountKey.json
   FIREBASE_PROJECT_ID=your-project-id
   ```

**Features**:
- User authentication (tokens)
- Firestore database for try-on history
- Automatic user profile creation

---

### 3. Cloudinary (Image Hosting)

**Purpose**: Cloud storage and CDN for try-on results

**Setup**:
1. Create account at https://cloudinary.com
2. Get credentials from dashboard
3. Add to `.env`:
   ```env
   CLOUDINARY_CLOUD_NAME=your-cloud-name
   CLOUDINARY_API_KEY=123456789012345
   CLOUDINARY_API_SECRET=xxxxxxxxxxxxxxxxxxxx
   ```

**Features**:
- Automatic image optimization
- CDN delivery
- Organized folder structure

---

## 💻 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run tests
pytest

# Run with coverage
pytest --cov=app tests/
```

### Code Quality

```bash
# Install linting tools
pip install black flake8 mypy

# Format code
black app/

# Lint
flake8 app/

# Type checking
mypy app/
```

### Development Tips

1. **Enable Debug Mode**: Set `DEBUG=true` in `.env` for auto-reload
2. **View Logs**: All safety checks log to console
3. **Test Safety**: Use `/api/validate` endpoint for quick testing
4. **API Docs**: Interactive testing at http://localhost:8000/docs

---

## 🚢 Deployment

### 🚀 Quick Deploy

Your backend is **ready to deploy** to Render or Railway! All configuration files are already created.

#### ⚡ Deploy to Render (1-Click)
1. Go to [render.com](https://render.com) and sign up/login
2. Click **"New +"** → **"Blueprint"**
3. Connect your repository
4. Render auto-detects `render.yaml` → Click **"Apply"**
5. Done! 🎉

#### ⚡ Deploy to Railway (1-Command)
```bash
# Install Railway CLI
npm i -g @railway/cli

# Deploy
railway login
railway init
railway up
```

### 📖 Detailed Deployment Guide

For complete step-by-step instructions, troubleshooting, and production configuration:

👉 **See [DEPLOYMENT.md](./DEPLOYMENT.md)** for the full deployment guide

Includes:
- ✅ Step-by-step Render deployment
- ✅ Step-by-step Railway deployment
- ✅ Environment variable configuration
- ✅ CORS setup for frontend
- ✅ Troubleshooting common issues
- ✅ Production best practices
- ✅ Cost estimates and scaling

### Production Checklist

- [ ] Set `DEBUG=false` in production
- [ ] Configure `CORS_ORIGINS` with your frontend URL
- [ ] Add API keys (Replicate, Firebase, Cloudinary) if using
- [ ] Test health endpoint: `/health`
- [ ] Verify try-on endpoint works
- [ ] Set up monitoring and logging
- [ ] Test all safety checks thoroughly

### Deployment Files Created

All required files are already in your backend:
- ✅ `Procfile` - Start command for both platforms
- ✅ `render.yaml` - Render configuration
- ✅ `railway.toml` - Railway configuration
- ✅ `runtime.txt` - Python version (3.9.0)
- ✅ `requirements.txt` - Updated with deployment-ready dependencies



---

## 📝 License

This project is part of the Stylora application. All rights reserved.

---

## 🤝 Contributing

1. Follow FastAPI best practices
2. Write comprehensive docstrings
3. Add tests for new features
4. Update this README for significant changes
5. Ensure all safety checks are thoroughly tested

---

## 📞 Support

For issues or questions:
1. Check the `/docs` endpoint for API documentation
2. Review safety check logs in console output
3. Verify environment configuration
4. Check health endpoint: `/health`

---

## 🎯 Quick Start Summary

```bash
# 1. Setup
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Configure
copy .env.example .env
# Edit .env with your settings

# 3. Run
uvicorn app.main:app --reload

# 4. Test
# Open: http://localhost:8000/docs
# Try: POST /api/tryon with test images
```

**That's it!** Your Stylora backend is ready to process virtual try-on requests with comprehensive safety validation. 🎉
