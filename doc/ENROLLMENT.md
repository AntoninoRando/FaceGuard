# Enrollment

The `src/api.py->add_probe_to_gallery` function enrolls a new face image into the biometric gallery. Here's how it works:

## Workflow
1. **Load Existing Gallery**
    - Loads the current gallery with encrypted embeddings from disk
    - Decrypts them for processing
    - If no gallery exists, creates a new empty one
2. **Validate Class ID**
    - Checks if class_id is new (not in existing labels) or existing
    - For new classes: requires person_name parameter (raises error if missing)
    - For existing classes: just adds another sample for that person
3. **Process the Probe Image**
    - **Face Detection**: Uses crop_face with [MTCNN](#mtcnn) to detect and crop the face to 160×160
    - **Degradation**: Applies random quality reduction (JPEG compression) and Gaussian noise to simulate real-world conditions
    - **quality_factor**: 0.6-1.0 (default random)
    - **noise_factor**: 0.0-0.04 (default random)
    - **Embedding Generation**: Uses FaceNet (InceptionResnetV1) to create a 512-dimensional embedding vector
4. **Update Gallery**
    - Appends the new embedding to the existing embeddings array
    - Adds the class_id to labels array
    - Stores the image path for reference
5. **Update Name-to-ID Mapping (New Class Only)**
    - If it's a new class, adds an entry to data/name_to_id.csv
    - Maps class_id → celebrity_name (person_name)
    - Sorts by class_id and saves
6. **Encrypt & Save**
    - Re-encrypts all embeddings using Fernet encryption
    - Saves the updated gallery to disk as a pickle file
    - Prints summary statistics
    - Key Features
    - **Encryption**: All embeddings are encrypted at rest for security
    - **Robustness**: Random degradation makes the system tolerant to image quality variations
    - **Flexibility**: Can add to existing identities or enroll new ones
    - **Validation**: Ensures face detection succeeds before adding to gallery

## MTCNN

MTCNN (Multi-Task Cascaded Convolutional Networks) is a powerful deep learning framework for face detection and alignment, using three cascaded Convolutional Neural Networks (P-Net, R-Net, O-Net) to efficiently identify faces, locate facial landmarks (eyes, nose, mouth), and refine bounding boxes across various scales and conditions. It excels at handling variations in lighting, expression, and occlusion, offering high accuracy for both locating faces and aligning them for further analysis, with popular Python implementations available for practical use. 