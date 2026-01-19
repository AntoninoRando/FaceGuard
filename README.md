# BioSys Face Recognition System

A complete face recognition system with identification (1:N), verification (1:1), anti-spoofing capabilities, a RESTful API with Swagger documentation, and a modern web interface.

## 🌐 NEW: Web Application!

**Complete web-based interface with video recording and real-time identification!**

### Quick Start
```bash
cd src
python api.py
```

Then open: **http://localhost:8000**

Features:
- 📝 **User Registration** - Capture and register faces
- 🎥 **Video Recording** - Record yourself for identification  
- 🛡️ **Anti-Spoofing** - Automatic liveness detection
- 🔍 **Identification** - Real-time face recognition
- 👥 **Gallery View** - Browse registered users
- 📊 **Performance Metrics** - Comprehensive biometric effectiveness indicators

**[→ Web App Quick Start Guide](WEBAPP_QUICKSTART.md)**

---

## 🚀 Quick Start

### REST API Server
Start the REST API server with Swagger documentation:
```bash
cd src
./start_api.sh
```

Then access:
- **Web App**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API**: http://localhost:8000/api

For detailed API documentation, see [API_README.md](API_README.md)

### Traditional Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

### Data Folder
- `data/initial_samples/`: Training images organized by class ID
  - Each subfolder is a class ID (e.g., `0/`, `1/`, `2/`)
  - Contains face images for that person
- `data/probes_for_test/`: Test images for evaluation
  - Same structure as `initial_samples/`
- `data/name_to_id.csv`: Maps class IDs to person names

### Embeddings Folder
- `embeddings/gallery.pkl`: Encrypted gallery of face embeddings
- `embeddings/gallery_key.key`: Encryption key (keep secure)

### Source Files
- `api.py`: RESTful API with FastAPI and Swagger (NEW!)
- `app.py`: GUI application with Tkinter
- `identification.py`: 1:N identification logic
- `verification.py`: 1:1 verification logic
- `antispoofing.py`: Presentation attack detection
- `sample_utils.py`: Image processing utilities
- `config.py`: Configuration settings
- `test_api_client.py`: Example API client (NEW!)

## Usage

### 🌐 REST API (Recommended)

**Start the server:**
```bash
cd src
./start_api.sh
```

**Test with the provided client:**
```bash
cd src
python test_api_client.py
```

**Use the API in your code:**
```python
import requests

# Identify a person
with open("face.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/identify",
        files={"image": f}
    )
    print(response.json())

# Verify identity
with open("face.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/verify",
        files={"image": f},
        data={"claimed_id": 5}
    )
    print(response.json())

# Detect spoofing
with open("face.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/antispoofing/detect",
        files={"image": f}
    )
    print(response.json())
```

See [API_README.md](API_README.md) for complete API documentation.

### Initialize Gallery
Before using the system, generate the encrypted gallery:
```bash
python initialize_gallery.py
```
This processes all images in `data/initial_samples/` and creates the encrypted gallery.

### Run Evaluation
**Identification (1:N):**
```bash
python identification.py
```

**Verification (1:1):**
```bash
python verification.py
```

**Anti-Spoofing:**
```bash
python antispoofing.py <video_file>
```

**Performance Metrics Evaluation:**
```bash
python evaluate_metrics.py --gallery_path data/initial_samples --test_path data/probes_for_test
```

This will generate:
- ROC Curve (Receiver Operating Characteristic)
- DET Curve (Detection Error Tradeoff)
- CMC Curve (Cumulative Match Characteristic)
- Comprehensive metrics report (JSON)

## 📊 Performance Metrics

The system provides comprehensive biometric effectiveness evaluation:

### Essential Metrics Displayed:
- **FAR (False Acceptance Rate)** - Rate of incorrectly accepting impostors
- **FRR (False Rejection Rate)** - Rate of incorrectly rejecting genuine users
- **EER (Equal Error Rate)** - Operating point where FAR = FRR
- **Accuracy** - Overall correctness of the system
- **Precision & Recall** - Classification performance indicators
- **ROC Curve** - True Positive Rate vs False Positive Rate
- **DET Curve** - FAR vs FRR tradeoff visualization
- **CMC Curve** - Identification rate at different ranks
- **Confusion Matrix** - Classification breakdown
- **Score Distribution** - Genuine vs impostor score separation
- **Anti-Spoofing Performance** - Liveness detection accuracy

### Accessing Metrics:

**Web Interface:**
Navigate to the "Metrics" tab in the web application at http://localhost:8000

**Command Line Evaluation:**
```bash
python evaluate_metrics.py
```

**API Endpoint:**
```bash
curl http://localhost:8000/metrics
```

The metrics help you understand:
- System reliability and accuracy
- Trade-offs between security (FAR) and usability (FRR)
- Optimal operating thresholds
- Performance across different ranks (identification)
- Anti-spoofing effectiveness

### GUI Application
Launch the interactive interface:
```bash
python app.py
```

Features:
- Webcam capture or file upload
- Identification mode: Find who the person is
- Verification mode: Verify claimed identity

## 🔌 API Features

The REST API provides:
- **Health Check**: Monitor system status
- **Identification (1:N)**: Identify a person from the gallery
- **Verification (1:1)**: Verify a claimed identity
- **Anti-Spoofing**: Detect presentation attacks
- **Gallery Management**: View identities and statistics
- **Swagger UI**: Interactive API documentation
- **OpenAPI Schema**: Standard API specification

All endpoints support:
- File upload via multipart/form-data
- Configurable thresholds and parameters
- Detailed JSON responses
- Error handling with meaningful messages

## Configuration

Edit `config.py` to adjust:
- Thresholds for identification/verification
- Data paths
- Embeddings storage location

## 📚 Additional Documentation

- [API_README.md](API_README.md) - Complete REST API documentation
- [requirements.txt](requirements.txt) - Python dependencies
- [src/test_api_client.py](src/test_api_client.py) - Example API usage

## 🔒 Security

- Gallery embeddings are encrypted using Fernet (symmetric encryption)
- Temporary files are automatically cleaned up
- Consider adding authentication for production use

## 📦 Dependencies

Key libraries:
- **FastAPI**: Modern web framework with automatic API docs
- **Uvicorn**: ASGI server
- **FaceNet (PyTorch)**: Face recognition model
- **MTCNN**: Face detection
- **OpenCV**: Image processing
- **Cryptography**: Gallery encryption
