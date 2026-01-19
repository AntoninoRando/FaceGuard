"""
RESTful API for Thermal Biometric System
Provides endpoints for face identification, verification, and anti-spoofing
with automatic Swagger/OpenAPI documentation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query, Path
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import tempfile
import os
import numpy as np
from pathlib import Path as FilePath
import shutil

# Import backend functions
from config import IDENTIFICATION_THRESHOLD, VERIFICATION_THRESHOLD, GALLERY_PATH
from identification import identify_probe_open_set, identify_probe_closed_set, process_probe_image
from verification import verify_claim, compute_claim_score
from sample_utils import (
    load_gallery, crop_face, face_to_embedding, add_probe_to_gallery,
    remove_identity_from_gallery, get_next_available_class_id
)
from antispoofing_detector import AntiSpoofingDetector
from facenet_pytorch import InceptionResnetV1
import pandas as pd

# Initialize FastAPI app
app = FastAPI(
    title="Thermal Biometric API",
    description="""
    RESTful API for facial biometric operations including:
    * **Identification (1:N)**: Identify a person from a gallery
    * **Verification (1:1)**: Verify a claimed identity
    * **Anti-Spoofing**: Detect presentation attacks
    
    All endpoints support image file uploads and return detailed results.
    """,
    version="1.0.0",
    contact={
        "name": "BioSys Team",
        "email": "support@biosys.example.com"
    },
    license_info={
        "name": "MIT",
    }
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = FilePath(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global variables for loaded models and gallery
model: Optional[InceptionResnetV1] = None
gallery: Optional[Dict] = None
antispoofing_detector: Optional[AntiSpoofingDetector] = None
classes_map: Dict[int, str] = {}


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class HealthResponse(BaseModel):
    status: str = Field(..., description="System status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    gallery_loaded: bool = Field(..., description="Whether the gallery is loaded")
    gallery_size: int = Field(..., description="Number of identities in gallery")

class IdentificationResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    predicted_label: Optional[int] = Field(None, description="Predicted identity label (-1 if unknown)")
    predicted_name: Optional[str] = Field(None, description="Name of predicted identity")
    min_distance: float = Field(..., description="Minimum distance to gallery")
    rejected: bool = Field(..., description="Whether probe was rejected as unknown")
    top_matches: List[Dict[str, Any]] = Field(..., description="Top K matching identities")
    threshold_used: float = Field(..., description="Threshold used for decision")
    error: Optional[str] = Field(None, description="Error message if any")

class VerificationResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    accepted: bool = Field(..., description="Whether verification passed")
    claimed_id: int = Field(..., description="Claimed identity")
    claimed_name: Optional[str] = Field(None, description="Name of claimed identity")
    distance: float = Field(..., description="Distance to claimed identity")
    threshold_used: float = Field(..., description="Threshold used for decision")
    confidence: float = Field(..., description="Confidence score (0-1)")
    gallery_updated: bool = Field(False, description="Whether gallery was updated")
    error: Optional[str] = Field(None, description="Error message if any")

class AntiSpoofingResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    is_live: bool = Field(..., description="Whether the face is live (not spoofed)")
    spoof_score: float = Field(..., description="Overall spoof score (0-1, higher is more likely spoofed)")
    confidence: float = Field(..., description="Confidence in the decision (0-1)")
    details: Dict[str, Any] = Field(..., description="Detailed analysis metrics")
    warning: Optional[str] = Field(None, description="Warning message")
    error: Optional[str] = Field(None, description="Error message if any")

class GalleryInfo(BaseModel):
    total_identities: int = Field(..., description="Number of unique identities")
    total_embeddings: int = Field(..., description="Total number of embeddings")
    identity_names: Dict[int, str] = Field(..., description="Mapping of IDs to names")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")

class RegistrationResponse(BaseModel):
    success: bool = Field(..., description="Whether registration was successful")
    class_id: Optional[int] = Field(None, description="Assigned class ID for the new identity")
    person_name: Optional[str] = Field(None, description="Name of the registered person")
    embeddings_added: int = Field(0, description="Number of embeddings added")
    total_embeddings: int = Field(0, description="Total embeddings in gallery after registration")
    total_classes: int = Field(0, description="Total classes in gallery after registration")
    message: Optional[str] = Field(None, description="Success message")
    error: Optional[str] = Field(None, description="Error message if any")

class UnregistrationResponse(BaseModel):
    success: bool = Field(..., description="Whether unregistration was successful")
    class_id: Optional[int] = Field(None, description="Class ID that was removed")
    person_name: Optional[str] = Field(None, description="Name of the removed person")
    embeddings_removed: int = Field(0, description="Number of embeddings removed")
    total_embeddings: int = Field(0, description="Total embeddings in gallery after removal")
    total_classes: int = Field(0, description="Total classes in gallery after removal")
    message: Optional[str] = Field(None, description="Success message")
    error: Optional[str] = Field(None, description="Error message if any")


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models and gallery on startup"""
    global model, gallery, antispoofing_detector, classes_map
    
    try:
        print("Loading FaceNet model...")
        model = InceptionResnetV1(pretrained='vggface2').eval()
        
        print("Loading gallery...")
        gallery = load_gallery()
        
        print("Initializing anti-spoofing detector...")
        antispoofing_detector = AntiSpoofingDetector()
        
        # Load names mapping
        try:
            df = pd.read_csv("data/name_to_id.csv")
            classes_map = dict(zip(df.class_id, df.celebrity_name))
        except Exception as e:
            print(f"Could not load name mapping: {e}")
            if gallery:
                unique_labels = np.unique(gallery['labels'])
                for lbl in unique_labels:
                    classes_map[lbl] = f"Person {lbl}"
        
        print("✓ System ready!")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down...")


# ============================================================================
# Helper Functions
# ============================================================================

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location"""
    try:
        suffix = FilePath(upload_file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            return tmp.name
    finally:
        upload_file.file.close()


def cleanup_temp_file(file_path: str):
    """Remove temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Warning: Could not delete temp file {file_path}: {e}")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["System"], include_in_schema=False)
async def root():
    """Serve the frontend application"""
    static_dir = FilePath(__file__).parent / "static"
    index_file = static_dir / "index.html"
    
    if index_file.exists():
        return FileResponse(index_file)
    else:
        return {
            "message": "Thermal Biometric API",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc",
            "frontend": "Frontend not found. Please check static folder."
        }

@app.get("/api", tags=["System"])
async def api_info():
    """API information endpoint"""
    return {
        "message": "Thermal Biometric API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check system health and status
    
    Returns the status of loaded models and gallery information
    """
    gallery_size = 0
    if gallery is not None:
        gallery_size = len(np.unique(gallery['labels']))
    
    return HealthResponse(
        status="operational" if model and gallery else "degraded",
        model_loaded=model is not None,
        gallery_loaded=gallery is not None,
        gallery_size=gallery_size
    )


@app.get("/gallery/info", response_model=GalleryInfo, tags=["Gallery"])
async def get_gallery_info():
    """
    Get information about the gallery
    
    Returns statistics about stored identities and their names
    """
    if gallery is None:
        raise HTTPException(status_code=503, detail="Gallery not loaded")
    
    unique_labels = np.unique(gallery['labels'])
    
    return GalleryInfo(
        total_identities=len(unique_labels),
        total_embeddings=len(gallery['embeddings']),
        identity_names=classes_map
    )


@app.post("/identify", response_model=IdentificationResponse, tags=["Identification"])
async def identify_face(
    image: UploadFile = File(..., description="Face image to identify"),
    threshold: Optional[float] = Query(
        None,
        description="Distance threshold for rejection (uses default if not provided)",
        ge=0.0,
        le=1.0
    ),
    top_k: int = Query(
        5,
        description="Number of top matches to return",
        ge=1,
        le=20
    ),
    metric: str = Query(
        "cosine",
        description="Distance metric to use",
        regex="^(cosine|euclidean)$"
    )
):
    """
    **Identify a person from the gallery (1:N matching)**
    
    Performs open-set identification:
    - Returns the most likely identity from the gallery
    - Rejects the probe as "unknown" if distance exceeds threshold
    
    **Parameters:**
    - **image**: Face image file (JPEG, PNG)
    - **threshold**: Distance threshold (default from config)
    - **top_k**: Number of top matches to return
    - **metric**: Distance metric ('cosine' or 'euclidean')
    
    **Returns:**
    - Predicted identity with confidence
    - Top K matching identities
    - Rejection status if probe is unknown
    """
    if model is None or gallery is None:
        raise HTTPException(status_code=503, detail="System not ready")
    
    temp_path = None
    try:
        # Save uploaded file
        temp_path = save_upload_file_tmp(image)
        
        # Process image
        face_img = crop_face(temp_path)
        if face_img is None:
            return IdentificationResponse(
                success=False,
                predicted_label=None,
                min_distance=999.0,  # Use large number instead of inf for JSON serialization
                rejected=True,
                top_matches=[],
                threshold_used=threshold or IDENTIFICATION_THRESHOLD,
                error="No face detected in image"
            )
        
        # Get embedding
        probe_emb = face_to_embedding(face_img, model=model)
        
        # Use threshold from config if not provided
        threshold_value = threshold if threshold is not None else IDENTIFICATION_THRESHOLD
        
        # Identify
        result = identify_probe_open_set(
            probe_emb,
            gallery,
            threshold_value,
            metric=metric,
            top_k=top_k
        )
        
        # Build top matches with names
        top_matches = []
        for i, (label, dist) in enumerate(zip(result['ranked_labels'], result['distances'])):
            top_matches.append({
                "rank": i + 1,
                "label": int(label),
                "name": classes_map.get(int(label), f"Person {label}"),
                "distance": float(dist),
                "confidence": max(0.0, 1.0 - float(dist))
            })
        
        predicted_name = None
        if not result['rejected']:
            predicted_name = classes_map.get(
                int(result['predicted_label']),
                f"Person {result['predicted_label']}"
            )
        
        return IdentificationResponse(
            success=True,
            predicted_label=int(result['predicted_label']) if not result['rejected'] else -1,
            predicted_name=predicted_name,
            min_distance=float(result['min_distance']),
            rejected=result['rejected'],
            top_matches=top_matches,
            threshold_used=threshold_value
        )
        
    except Exception as e:
        return IdentificationResponse(
            success=False,
            predicted_label=None,
            min_distance=999.0,  # Use large number instead of inf for JSON serialization
            rejected=True,
            top_matches=[],
            threshold_used=threshold or IDENTIFICATION_THRESHOLD,
            error=str(e)
        )
    finally:
        if temp_path:
            cleanup_temp_file(temp_path)


@app.post("/verify", response_model=VerificationResponse, tags=["Verification"])
async def verify_identity(
    image: UploadFile = File(..., description="Face image to verify"),
    claimed_id: int = Form(..., description="Claimed identity ID"),
    threshold: Optional[float] = Form(
        None,
        description="Distance threshold for acceptance (uses default if not provided)",
        ge=0.0,
        le=1.0
    ),
    update_gallery: bool = Form(
        False,
        description="Whether to add accepted probe to gallery (adaptive learning)"
    ),
    metric: str = Form(
        "cosine",
        description="Distance metric to use (cosine or euclidean)"
    )
):
    """
    **Verify a claimed identity (1:1 matching)**
    
    Verifies whether the provided image matches the claimed identity:
    - Compares probe against stored templates of claimed identity
    - Accepts if distance is below threshold
    
    **Parameters:**
    - **image**: Face image file (JPEG, PNG)
    - **claimed_id**: Identity ID being claimed (from gallery)
    - **threshold**: Distance threshold for acceptance (default from config)
    - **update_gallery**: Add accepted probe to gallery for adaptive learning
    - **metric**: Distance metric ('cosine' or 'euclidean')
    
    **Returns:**
    - Verification decision (accepted/rejected)
    - Distance to claimed identity
    - Confidence score
    """
    if model is None or gallery is None:
        raise HTTPException(status_code=503, detail="System not ready")
    
    temp_path = None
    try:
        # Save uploaded file
        temp_path = save_upload_file_tmp(image)
        
        # Use threshold from config if not provided
        threshold_value = threshold if threshold is not None else VERIFICATION_THRESHOLD
        
        # Perform verification
        result = verify_claim(
            temp_path,
            claimed_id,
            threshold=threshold_value,
            gallery_path=GALLERY_PATH,
            model=model,
            update_gallery=update_gallery,
            metric=metric
        )
        
        if 'error' in result:
            return VerificationResponse(
                success=False,
                accepted=False,
                claimed_id=claimed_id,
                claimed_name=classes_map.get(claimed_id, f"Person {claimed_id}"),
                distance=999.0,  # Use large number instead of inf for JSON serialization
                threshold_used=threshold_value,
                confidence=0.0,
                error=result['error']
            )
        
        # Calculate confidence (inverse of normalized distance)
        confidence = max(0.0, 1.0 - (result['distance'] / (threshold_value * 2)))
        
        return VerificationResponse(
            success=True,
            accepted=result['accepted'],
            claimed_id=claimed_id,
            claimed_name=classes_map.get(claimed_id, f"Person {claimed_id}"),
            distance=float(result['distance']),
            threshold_used=threshold_value,
            confidence=confidence,
            gallery_updated=result.get('gallery_updated', False)
        )
        
    except Exception as e:
        return VerificationResponse(
            success=False,
            accepted=False,
            claimed_id=claimed_id,
            claimed_name=classes_map.get(claimed_id, f"Person {claimed_id}"),
            distance=999.0,  # Use large number instead of inf for JSON serialization
            threshold_used=threshold if threshold is not None else VERIFICATION_THRESHOLD,
            confidence=0.0,
            error=str(e)
        )
    finally:
        if temp_path:
            cleanup_temp_file(temp_path)


@app.post("/antispoofing/detect", response_model=AntiSpoofingResponse, tags=["Anti-Spoofing"])
async def detect_spoofing(
    file: UploadFile = File(..., description="Face image or video file to check for spoofing")
):
    """
    **Detect presentation attacks (spoofing)**
    
    Analyzes an image or video to determine if it contains a live face or a spoofing attempt:
    - Printed photos
    - Video replay attacks
    - Masks
    
    Uses multiple detection methods:
    - Texture analysis (LBP)
    - Frequency domain analysis (moiré patterns)
    - Motion patterns (video only)
    - Eye blink detection (video only)
    - Liveness indicators
    
    **Parameters:**
    - **file**: Face image (JPEG, PNG) or video file (MP4, AVI, MOV, WEBM)
    
    **Returns:**
    - Liveness decision (live/spoofed)
    - Spoof score and confidence
    - Detailed analysis metrics
    
    **Note:** Video analysis is more accurate as it includes motion and blink detection.
    """
    if antispoofing_detector is None:
        raise HTTPException(status_code=503, detail="Anti-spoofing detector not ready")
    
    temp_path = None
    try:
        # Save uploaded file
        temp_path = save_upload_file_tmp(file)
        
        # Determine file type and perform appropriate analysis
        file_extension = FilePath(file.filename).suffix.lower()
        video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv']
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        if file_extension in video_extensions:
            # Perform video analysis
            result = antispoofing_detector.analyze_video(temp_path)
        elif file_extension in image_extensions:
            # Perform image analysis
            result = antispoofing_detector.analyze_single_image(temp_path)
        else:
            return AntiSpoofingResponse(
                success=False,
                is_live=False,
                spoof_score=1.0,
                confidence=0.0,
                details={},
                error=f"Unsupported file type: {file_extension}. Supported: {', '.join(image_extensions + video_extensions)}"
            )
        
        if 'error' in result:
            return AntiSpoofingResponse(
                success=False,
                is_live=False,
                spoof_score=1.0,
                confidence=0.0,
                details={},
                error=result['error']
            )
        
        return AntiSpoofingResponse(
            success=True,
            is_live=result['is_live'],
            spoof_score=float(result['spoof_score']),
            confidence=float(result['confidence']),
            details=result.get('details', {}),
            warning=result.get('warning')
        )
        
    except Exception as e:
        return AntiSpoofingResponse(
            success=False,
            is_live=False,
            spoof_score=1.0,
            confidence=0.0,
            details={},
            error=str(e)
        )
    finally:
        if temp_path:
            cleanup_temp_file(temp_path)


@app.get("/identities", tags=["Gallery"])
async def list_identities():
    """
    **List all identities in the gallery**
    
    Returns a list of all registered identities with their IDs and names
    """
    if gallery is None:
        raise HTTPException(status_code=503, detail="Gallery not loaded")
    
    unique_labels = np.unique(gallery['labels'])
    identities = []
    
    for label in unique_labels:
        count = np.sum(gallery['labels'] == label)
        identities.append({
            "id": int(label),
            "name": classes_map.get(int(label), f"Person {label}"),
            "sample_count": int(count)
        })
    
    return {
        "total": len(identities),
        "identities": identities
    }


@app.post("/register", response_model=RegistrationResponse, tags=["Gallery"])
async def register_identity(
    image: UploadFile = File(..., description="Face image to register"),
    name: str = Form(..., description="Name of the person to register"),
    class_id: Optional[int] = Form(None, description="Specific class ID (optional, auto-assigned if not provided)")
):
    """
    **Register a new identity in the system**
    
    Adds a new person to the gallery permanently:
    - Detects and crops the face from the image
    - Generates embedding and stores it
    - Assigns a unique class ID (auto or manual)
    - Updates the name mapping file
    - Reloads the gallery in memory
    
    **Parameters:**
    - **image**: Face image file (JPEG, PNG)
    - **name**: Full name of the person
    - **class_id**: (Optional) Specific ID to assign. If not provided, next available ID is used
    
    **Returns:**
    - Registration status with assigned class ID
    - Updated gallery statistics
    
    **Note:** This operation permanently modifies the gallery file and name mapping.
    """
    global gallery, classes_map
    
    if model is None:
        raise HTTPException(status_code=503, detail="System not ready")
    
    temp_path = None
    try:
        # Save uploaded file
        temp_path = save_upload_file_tmp(image)
        
        # Determine class_id
        if class_id is None:
            class_id = get_next_available_class_id()
        else:
            # Check if class_id already exists
            if gallery is not None and class_id in gallery['labels']:
                return RegistrationResponse(
                    success=False,
                    error=f"Class ID {class_id} already exists. Use a different ID or omit to auto-assign."
                )
        
        # Add to gallery
        result_gallery = add_probe_to_gallery(
            image_path=temp_path,
            class_id=class_id,
            gallery_path=GALLERY_PATH,
            model=model,
            person_name=name
        )
        
        # Reload gallery in memory
        gallery = load_gallery(GALLERY_PATH)
        
        # Update classes_map
        classes_map[class_id] = name
        
        # Get statistics
        total_embeddings = len(gallery['embeddings'])
        total_classes = len(np.unique(gallery['labels']))
        embeddings_for_this_class = np.sum(gallery['labels'] == class_id)
        
        return RegistrationResponse(
            success=True,
            class_id=class_id,
            person_name=name,
            embeddings_added=1,
            total_embeddings=total_embeddings,
            total_classes=total_classes,
            message=f"Successfully registered '{name}' with class ID {class_id}"
        )
        
    except ValueError as e:
        return RegistrationResponse(
            success=False,
            error=str(e)
        )
    except Exception as e:
        return RegistrationResponse(
            success=False,
            error=f"Registration failed: {str(e)}"
        )
    finally:
        if temp_path:
            cleanup_temp_file(temp_path)


@app.delete("/unregister/{identity_id}", response_model=UnregistrationResponse, tags=["Gallery"])
async def unregister_identity(
    identity_id: int = Path(..., description="Identity ID to remove from the system")
):
    """
    **Unregister an identity from the system**
    
    Removes all embeddings and data for a specific identity:
    - Deletes all embeddings for the identity
    - Removes entry from name mapping file
    - Updates the gallery file
    - Reloads the gallery in memory
    
    **Parameters:**
    - **identity_id**: The class ID of the identity to remove
    
    **Returns:**
    - Unregistration status
    - Number of embeddings removed
    - Updated gallery statistics
    
    **Note:** This operation permanently modifies the gallery file and name mapping.
    **Warning:** This action cannot be undone!
    """
    global gallery, classes_map
    
    if gallery is None:
        raise HTTPException(status_code=503, detail="Gallery not loaded")
    
    try:
        # Check if identity exists
        if identity_id not in gallery['labels']:
            return UnregistrationResponse(
                success=False,
                error=f"Identity ID {identity_id} not found in gallery"
            )
        
        # Get person name before removal
        person_name = classes_map.get(identity_id, f"Person {identity_id}")
        
        # Remove from gallery
        result = remove_identity_from_gallery(
            class_id=identity_id,
            gallery_path=GALLERY_PATH
        )
        
        if not result['success']:
            return UnregistrationResponse(
                success=False,
                error=result.get('error', 'Unknown error during unregistration')
            )
        
        # Reload gallery in memory
        gallery = load_gallery(GALLERY_PATH)
        
        # Update classes_map
        if identity_id in classes_map:
            del classes_map[identity_id]
        
        return UnregistrationResponse(
            success=True,
            class_id=identity_id,
            person_name=result.get('person_name', person_name),
            embeddings_removed=result['embeddings_removed'],
            total_embeddings=result['total_embeddings'],
            total_classes=result['total_classes'],
            message=f"Successfully unregistered '{result.get('person_name', person_name)}' (ID: {identity_id})"
        )
        
    except Exception as e:
        return UnregistrationResponse(
            success=False,
            error=f"Unregistration failed: {str(e)}"
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom handler for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Custom handler for general exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
