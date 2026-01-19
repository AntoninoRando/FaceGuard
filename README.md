# BioSys Face Recognition System

A complete face recognition system with identification, verification, anti-spoofing capabilities, a RESTful API with Swagger documentation, and a web interface.

## Quick Start

Before using the system, generate the **encrypted gallery**:
```bash
python initialize_gallery.py
```
This processes all images in `data/initial_samples/` and creates the encrypted gallery.


### Web Application

```bash
cd src
python api.py
```

Then open: **http://localhost:8000**

Features:
- **User Registration** - Capture and register faces
-**Video Recording** - Record yourself for identification  
- **Anti-Spoofing** - Automatic liveness detection
- **Identification** - Real-time face recognition
- **Gallery View** - Browse registered users
- **Performance Metrics** - Comprehensive biometric effectiveness indicators


### REST API Server

```bash
cd src
python api.py
```

Then access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API**: http://localhost:8000/api



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
- `api.py`: RESTful API with FastAPI and Swagger
- `app.py`: GUI application with Tkinter
- `identification.py`: 1:N identification logic
- `verification.py`: 1:1 verification logic
- `antispoofing_detector.py`: Presentation attack detection
- `/antispoofing`: Directory with antispoofing detectors modules
- `sample_utils.py`: Image processing utilities
- `config.py`: Configuration settings

## System Assessment

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
python antispoofing_detector.py <video_file>
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

### Performance Metrics

The system provides comprehensive biometric effectiveness evaluation:

#### Essential Metrics Displayed:
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


### Configuration

Edit `config.py` to adjust:
- Thresholds for identification/verification
- Data paths
- Embeddings storage location
