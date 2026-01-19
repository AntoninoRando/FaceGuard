from PIL import Image
import numpy as np
from mtcnn import MTCNN
import io
import torch
from facenet_pytorch import InceptionResnetV1
import os
import pickle
from tqdm import tqdm
import pandas as pd
from cryptography.fernet import Fernet

from config import SAMPLES_PATH, ENCRYPTION_KEY_FILE, GALLERY_PATH, EMBEDDINGS_FOLDER

def get_encryption_key():
    """
    Get or generate encryption key for gallery embeddings.
    
    Returns:
        bytes: Encryption key
    """
    # Ensure embeddings folder exists
    os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)
    
    if os.path.exists(ENCRYPTION_KEY_FILE):
        # Load existing key
        with open(ENCRYPTION_KEY_FILE, 'rb') as f:
            key = f.read()
    else:
        # Generate new key
        key = Fernet.generate_key()
        with open(ENCRYPTION_KEY_FILE, 'wb') as f:
            f.write(key)
        print(f"Generated new encryption key: {ENCRYPTION_KEY_FILE}")
    return key

def encrypt_embeddings(embeddings):
    """
    Encrypt embedding array.
    
    Args:
        embeddings (numpy.ndarray): Array of embeddings to encrypt
    
    Returns:
        bytes: Encrypted embeddings
    """
    key = get_encryption_key()
    fernet = Fernet(key)
    
    # Serialize embeddings to bytes
    embeddings_bytes = pickle.dumps(embeddings)
    
    # Encrypt
    encrypted_data = fernet.encrypt(embeddings_bytes)
    
    return encrypted_data

def decrypt_embeddings(encrypted_data):
    """
    Decrypt embedding array.
    
    Args:
        encrypted_data (bytes): Encrypted embeddings
    
    Returns:
        numpy.ndarray: Decrypted embeddings array
    """
    key = get_encryption_key()
    fernet = Fernet(key)
    
    # Decrypt
    embeddings_bytes = fernet.decrypt(encrypted_data)
    
    # Deserialize
    embeddings = pickle.loads(embeddings_bytes)
    
    return embeddings

def crop_face(image_path, target_size=(160, 160), margin=20):
    """
    Detect and crop face from image.
    If multiple faces detected, takes the one with highest confidence.
    """
    try:
        img = Image.open(image_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # FIX: Resize huge images before detection to prevent MemoryError
        # MTCNN doesn't need 20k pixels to detect a face, 2048px is enough
        max_dimension = 2048
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
        detector = MTCNN()
        # Detect faces
        faces = detector.detect_faces(np.array(img))
        
        if not faces:
            # print(f"Warning: No face detected in {image_path}")
            return None
            
        # Take the result with highest confidence
        face = max(faces, key=lambda x: x['confidence'])
        box = face['box']
        
        # Add margin
        x, y, w, h = box
        img_w, img_h = img.size
        
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_w, x + w + margin)
        y2 = min(img_h, y + h + margin)
        
        # Crop
        face_img = img.crop((x1, y1, x2, y2))
        
        # Resize to target size
        face_img = face_img.resize(target_size, Image.Resampling.LANCZOS)
        
        return face_img
        
    except Exception as e:
        print(f"Error cropping face in {image_path}: {str(e)}")
        return None

def add_degradation(face_img, quality_factor=0.7, noise_factor=0.02):
    """
    Add quality degradation and noise to a face image to simulate real-world variations.
    This helps make biometric systems more robust to image quality variations.
    
    Args:
        face_img (PIL.Image): Input face image (typically from crop_face function)
        quality_factor (float): JPEG quality factor. Range: 0.5 to 1.0
            - 1.0 = no degradation (quality=95)
            - 0.9 = slight degradation (quality=85)
            - 0.7 = moderate degradation (quality=70) [RECOMMENDED]
            - 0.5 = noticeable degradation (quality=50)
        noise_factor (float): Gaussian noise intensity. Range: 0.0 to 0.05
            - 0.0 = no noise
            - 0.01 = slight noise [RECOMMENDED for subtle effect]
            - 0.02 = moderate noise [RECOMMENDED for training robustness]
            - 0.05 = heavy noise (may affect recognition)
    
    Returns:
        PIL.Image: Degraded face image with reduced quality and added noise
    """    
    # Convert quality_factor (0.5-1.0) to JPEG quality (50-95)
    jpeg_quality = int(50 + (quality_factor - 0.5) * 90)
    jpeg_quality = max(50, min(95, jpeg_quality))  # Clamp between 50 and 95
    
    # Apply JPEG compression to reduce quality
    buffer = io.BytesIO()
    face_img.save(buffer, format='JPEG', quality=jpeg_quality)
    buffer.seek(0)
    degraded_img = Image.open(buffer)
    
    # Convert to numpy array for noise addition
    img_array = np.array(degraded_img).astype(np.float32)
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_factor * 255, img_array.shape)
    noisy_array = img_array + noise
    
    # Clip values to valid range [0, 255]
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    result_img = Image.fromarray(noisy_array)
    
    return result_img

def face_to_embedding(face_img, model=None):
    """
    Convert a face image to a FaceNet embedding vector.
    
    Args:
        face_img (PIL.Image): Face image of size 160x160 (typically from crop_face function)
        model (InceptionResnetV1, optional): Pre-loaded FaceNet model. If None, loads pretrained model.
    
    Returns:
        numpy.ndarray: 512-dimensional embedding vector representing the face
    """
    # Load pretrained FaceNet model if not provided
    if model is None:
        model = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Convert PIL Image to tensor
    # FaceNet expects input normalized to [-1, 1] range
    img_array = np.array(face_img).astype(np.float32)
    
    # Convert from [H, W, C] to [C, H, W] and normalize to [0, 1]
    img_tensor = torch.from_numpy(img_array / 255.0).permute(2, 0, 1)
    
    # Normalize to [-1, 1] range (FaceNet standard)
    img_tensor = (img_tensor - 0.5) / 0.5
    
    # Add batch dimension [1, C, H, W]
    img_tensor = img_tensor.unsqueeze(0)
    
    # Generate embedding
    with torch.no_grad():
        embedding = model(img_tensor)
    
    # Convert to numpy array and remove batch dimension
    embedding_np = embedding.squeeze().numpy()
    
    return embedding_np

def initialize_gallery():
    """
    Initialize the gallery by processing all face images from SAMPLES_PATH.
    For each image:
    - Crop the face using MTCNN
    - Apply random degradation (quality and noise)
    - Generate FaceNet embedding
    - Store embedding with its class label
    
    Saves the gallery as 'gallery.pkl' containing:
    - 'embeddings': numpy array of shape (N, 512) with N face embeddings
    - 'labels': numpy array of shape (N,) with corresponding class IDs
    - 'image_paths': list of original image paths for reference
    
    Returns:
        dict: Gallery data with 'embeddings', 'labels', and 'image_paths'
    """
    print("Initializing gallery...")
    
    # Load FaceNet model once for efficiency
    print("Loading FaceNet model...")
    model = InceptionResnetV1(pretrained='vggface2').eval()
    
    embeddings_list = []
    labels_list = []
    image_paths_list = []
    
    # Dynamically detect classes in SAMPLES_PATH
    if not os.path.exists(SAMPLES_PATH):
        raise FileNotFoundError(f"Samples path check failed: {SAMPLES_PATH}")
        
    class_folders = [d for d in os.listdir(SAMPLES_PATH) 
                    if os.path.isdir(os.path.join(SAMPLES_PATH, d)) and d.isdigit()]
    class_ids = sorted([int(d) for d in class_folders])
    
    failed_images = []
    
    print(f"Found {len(class_ids)} classes: {class_ids}")
    
    for class_id in class_ids:
        class_folder = os.path.join(SAMPLES_PATH, str(class_id))
        
        # Get all image files in the class folder
        image_files = [f for f in os.listdir(class_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\nProcessing class {class_id}: {len(image_files)} images")
        
        for img_file in tqdm(image_files, desc=f"Class {class_id}"):
            img_path = os.path.join(class_folder, img_file)
            
            try:
                # Step 1: Crop face
                face_img = crop_face(img_path, target_size=(160, 160))
                
                if face_img is None:
                    failed_images.append(img_path)
                    continue
                
                # Step 2: Apply random degradation
                quality_factor = np.random.uniform(0.6, 1.0)
                noise_factor = np.random.uniform(0.0, 0.04)
                degraded_face = add_degradation(face_img, quality_factor, noise_factor)
                
                # Step 3: Generate embedding
                embedding = face_to_embedding(degraded_face, model=model)
                
                # Step 4: Store results
                embeddings_list.append(embedding)
                labels_list.append(class_id)
                image_paths_list.append(img_path)
                
            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
                failed_images.append(img_path)
    
    # Convert lists to numpy arrays
    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)
    
    # Encrypt embeddings before saving
    encrypted_embeddings = encrypt_embeddings(embeddings)
    
    # Create gallery dictionary with encrypted embeddings
    gallery = {
        'embeddings': encrypted_embeddings,
        'labels': labels,
        'image_paths': image_paths_list,
        'encrypted': True  # Flag to indicate embeddings are encrypted
    }
    
    # Ensure embeddings folder exists
    os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)
    
    # Save gallery to file
    with open(GALLERY_PATH, 'wb') as f:
        pickle.dump(gallery, f)
    
    print(f"\n{'='*60}")
    print(f"Gallery initialization complete!")
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Classes: {len(np.unique(labels))}")
    print(f"Failed images: {len(failed_images)}")
    print(f"Gallery saved to: {GALLERY_PATH}")
    print(f"{'='*60}\n")
    
    if failed_images:
        print(f"Failed images list:")
        for img in failed_images[:10]:  # Show first 10
            print(f"  - {img}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
    
    return gallery

def load_gallery(gallery_path=None):
    """
    Load the gallery from disk and decrypt embeddings if encrypted.
    
    Args:
        gallery_path (str): Path to the gallery file
    
    Returns:
        dict: Gallery data with decrypted 'embeddings', 'labels', and 'image_paths'
    """
    if gallery_path is None:
        gallery_path = GALLERY_PATH
    
    if not os.path.exists(gallery_path):
        raise FileNotFoundError(f"Gallery file not found: {gallery_path}")
    
    with open(gallery_path, 'rb') as f:
        gallery = pickle.load(f)
    
    # Decrypt embeddings if they are encrypted
    if gallery.get('encrypted', False):
        gallery['embeddings'] = decrypt_embeddings(gallery['embeddings'])
    
    return gallery

def add_probe_to_gallery(image_path, class_id, gallery_path=None, 
                         quality_factor=None, noise_factor=None, model=None, person_name=None):
    """
    Add a new probe image embedding to the gallery for an existing or new class.
    
    Args:
        image_path (str): Path to the probe image
        class_id (int): Class ID to assign (can be existing or new)
        gallery_path (str): Path to the gallery file
        quality_factor (float, optional): JPEG quality factor (0.5-1.0). If None, random [0.6, 1.0]
        noise_factor (float, optional): Noise intensity (0.0-0.05). If None, random [0.0, 0.04]
        model (InceptionResnetV1, optional): Pre-loaded FaceNet model. If None, loads model.
        person_name (str, optional): Name of the person. Required if adding a new class.
    
    Returns:
        dict: Updated gallery data
    """
    if gallery_path is None:
        gallery_path = GALLERY_PATH
    
    # Load existing gallery
    try:
        gallery = load_gallery(gallery_path)
        print(f"Loaded existing gallery with {len(gallery['embeddings'])} embeddings")
        existing_labels = gallery['labels']
    except FileNotFoundError:
        print("No existing gallery found. Creating new gallery.")
        gallery = {
            'embeddings': np.array([]),
            'labels': np.array([]),
            'image_paths': []
        }
        existing_labels = np.array([])
    
    # Check if it's a new class (before adding)
    is_new_class = class_id not in existing_labels
    
    # If new class, person_name is required
    if is_new_class and person_name is None:
        raise ValueError(f"person_name is required when adding a new class (class_id={class_id})")
    
    # Load FaceNet model if not provided
    if model is None:
        print("Loading FaceNet model...")
        model = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Process the probe image
    print(f"Processing probe image: {image_path}")
    
    # Step 1: Crop face
    face_img = crop_face(image_path, target_size=(160, 160))
    
    if face_img is None:
        raise ValueError(f"No face detected in {image_path}")
    
    # Step 2: Apply degradation (random if not specified)
    if quality_factor is None:
        quality_factor = np.random.uniform(0.6, 1.0)
    if noise_factor is None:
        noise_factor = np.random.uniform(0.0, 0.04)
    
    degraded_face = add_degradation(face_img, quality_factor, noise_factor)
    
    # Step 3: Generate embedding
    embedding = face_to_embedding(degraded_face, model=model)
    
    # Step 4: Add to gallery (embeddings are in decrypted form at this point)
    if len(gallery['embeddings']) == 0:
        # First embedding in gallery
        embeddings_array = embedding.reshape(1, -1)
        gallery['labels'] = np.array([class_id])
        gallery['image_paths'] = [image_path]
    else:
        # Append to existing gallery
        embeddings_array = np.vstack([gallery['embeddings'], embedding])
        gallery['labels'] = np.append(gallery['labels'], class_id)
        gallery['image_paths'].append(image_path)
    
    # Step 5: If new class, add to CSV
    if is_new_class:
        csv_path = 'data/name_to_id.csv'
        try:
            # Read existing CSV
            df = pd.read_csv(csv_path)
            
            # Add new entry
            new_row = pd.DataFrame({'class_id': [class_id], 'celebrity_name': [person_name]})
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Sort by class_id
            df = df.sort_values('class_id').reset_index(drop=True)
            
            # Save updated CSV
            df.to_csv(csv_path, index=False)
            print(f"✓ Added '{person_name}' (class_id={class_id}) to {csv_path}")
        except Exception as e:
            print(f"Warning: Could not update CSV file: {str(e)}")
    
    # Encrypt embeddings before saving
    encrypted_embeddings = encrypt_embeddings(embeddings_array)
    gallery['embeddings'] = encrypted_embeddings
    gallery['encrypted'] = True
    
    # Save updated gallery
    with open(gallery_path, 'wb') as f:
        pickle.dump(gallery, f)
    
    print(f"\n{'='*60}")
    if is_new_class:
        print(f"✓ Added probe to NEW class {class_id}: {person_name}")
    else:
        print(f"✓ Added probe to EXISTING class {class_id}")
    print(f"  Degradation: quality={quality_factor:.2f}, noise={noise_factor:.3f}")
    print(f"  Gallery now contains {len(gallery['embeddings'])} embeddings")
    print(f"  Total classes: {len(np.unique(gallery['labels']))}")
    print(f"  Gallery saved to: {gallery_path}")
    print(f"{'='*60}\n")
    
    return gallery
def remove_identity_from_gallery(class_id, gallery_path=None):
    """
    Remove all embeddings for a specific identity from the gallery.
    
    Args:
        class_id (int): Class ID to remove
        gallery_path (str, optional): Path to the gallery file
    
    Returns:
        dict: Updated gallery data or error dict
    """
    if gallery_path is None:
        gallery_path = GALLERY_PATH
    
    try:
        # Load existing gallery
        gallery = load_gallery(gallery_path)
        print(f"Loaded existing gallery with {len(gallery['embeddings'])} embeddings")
        
        # Check if class_id exists
        if class_id not in gallery['labels']:
            return {
                'success': False,
                'error': f"Class ID {class_id} not found in gallery"
            }
        
        # Count embeddings to remove
        embeddings_to_remove = np.sum(gallery['labels'] == class_id)
        
        # Filter out the class
        mask = gallery['labels'] != class_id
        new_embeddings = gallery['embeddings'][mask]
        new_labels = gallery['labels'][mask]
        new_image_paths = [path for i, path in enumerate(gallery['image_paths']) if mask[i]]
        
        # Encrypt embeddings before saving
        encrypted_embeddings = encrypt_embeddings(new_embeddings)
        
        # Create updated gallery
        updated_gallery = {
            'embeddings': encrypted_embeddings,
            'labels': new_labels,
            'image_paths': new_image_paths,
            'encrypted': True
        }
        
        # Save updated gallery
        with open(gallery_path, 'wb') as f:
            pickle.dump(updated_gallery, f)
        
        # Update CSV file - remove the entry
        csv_path = 'data/name_to_id.csv'
        person_name = None
        try:
            df = pd.read_csv(csv_path)
            # Get the name before removing
            person_row = df[df['class_id'] == class_id]
            if not person_row.empty:
                person_name = person_row['celebrity_name'].values[0]
            
            # Remove the row
            df = df[df['class_id'] != class_id]
            df.to_csv(csv_path, index=False)
            print(f"✓ Removed class_id={class_id} from {csv_path}")
        except Exception as e:
            print(f"Warning: Could not update CSV file: {str(e)}")
        
        print(f"\n{'='*60}")
        print(f"✓ Removed identity (class_id={class_id})")
        if person_name:
            print(f"  Name: {person_name}")
        print(f"  Embeddings removed: {embeddings_to_remove}")
        print(f"  Gallery now contains {len(new_embeddings)} embeddings")
        print(f"  Total classes: {len(np.unique(new_labels))}")
        print(f"  Gallery saved to: {gallery_path}")
        print(f"{'='*60}\n")
        
        return {
            'success': True,
            'class_id': class_id,
            'person_name': person_name,
            'embeddings_removed': int(embeddings_to_remove),
            'total_embeddings': len(new_embeddings),
            'total_classes': len(np.unique(new_labels))
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_next_available_class_id(gallery_path=None):
    """
    Get the next available class ID for a new identity.
    
    Args:
        gallery_path (str, optional): Path to the gallery file
    
    Returns:
        int: Next available class ID
    """
    if gallery_path is None:
        gallery_path = GALLERY_PATH
    
    try:
        # Load existing gallery
        gallery = load_gallery(gallery_path)
        existing_ids = np.unique(gallery['labels'])
        
        if len(existing_ids) == 0:
            return 0
        
        # Return max ID + 1
        return int(np.max(existing_ids) + 1)
        
    except FileNotFoundError:
        # No gallery exists yet
        return 0