"""
Face Verification System (1:1)
Handles verification of identity claims (Is this person who they claim to be?)
and evaluation of the verification system using FAR, FRR, EER, ROC, and DET metrics.
"""

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean

from sample_utils import crop_face, face_to_embedding, load_gallery, add_probe_to_gallery
from facenet_pytorch import InceptionResnetV1
from config import TEST_PATH, VERIFICATION_THRESHOLD, GALLERY_PATH

def get_class_embeddings(gallery, class_id):
    """
    Retrieve all embeddings belonging to a specific class from the gallery.
    
    Args:
        gallery (dict): Gallery data
        class_id (int): ID of the class
        
    Returns:
        numpy.ndarray: Array of embeddings for the class, or empty array if not found
    """
    indices = np.where(gallery['labels'] == class_id)[0]
    if len(indices) == 0:
        return np.array([])
    return gallery['embeddings'][indices]

def compute_claim_score(probe_embedding, claimed_class_id, gallery, metric='cosine'):
    """
    Compute the score (min distance) for a verification claim.
    
    Args:
        probe_embedding (numpy.ndarray): Probe embedding
        claimed_class_id (int): The identity being claimed
        gallery (dict): Gallery data
        metric (str): 'cosine' or 'euclidean'
        
    Returns:
        float: Minimum distance to the claimed class (lower is better match).
               Returns infinity if class has no embeddings.
    """
    class_embeddings = get_class_embeddings(gallery, claimed_class_id)
    
    if len(class_embeddings) == 0:
        return float('inf')
    
    distances = []
    if metric == 'cosine':
        for emb in class_embeddings:
            distances.append(cosine(probe_embedding, emb))
    elif metric == 'euclidean':
        for emb in class_embeddings:
            distances.append(euclidean(probe_embedding, emb))
            
    return np.min(distances)

def verify_claim(image_path, claimed_class_id, threshold=None, gallery_path=None, 
                 model=None, update_gallery=False, metric='cosine'):
    """
    Operational 1:1 Verification function.
    Verifies if the image belongs to the claimed identity.
    
    Args:
        image_path (str): Path to probe image
        claimed_class_id (int): ID of the identity being claimed
        threshold (float): Distance threshold for acceptance
        gallery_path (str): Path to gallery file
        model (InceptionResnetV1): Loaded model
        update_gallery (bool): If True, adds accepted probes to gallery (adaptive)
        metric (str): Distance metric
        
    Returns:
        dict: detailed result including decision, distance, and update status
    """
    # Use default threshold from config if not provided
    if threshold is None:
        threshold = VERIFICATION_THRESHOLD
    
    if gallery_path is None:
        gallery_path = GALLERY_PATH
    
    # Load gallery (decrypts automatically)
    gallery = load_gallery(gallery_path)
    
    # Init model if needed
    if model is None:
        model = InceptionResnetV1(pretrained='vggface2').eval()
        
    # Process probe
    face_img = crop_face(image_path)
    if face_img is None:
        return {'accepted': False, 'error': 'No face detected'}
        
    probe_emb = face_to_embedding(face_img, model=model)
    
    # Compute score
    min_distance = compute_claim_score(probe_emb, claimed_class_id, gallery, metric)
    
    # Decision
    accepted = min_distance <= threshold
    
    result = {
        'accepted': accepted,
        'distance': min_distance,
        'threshold': threshold,
        'claimed_id': claimed_class_id,
        'updated_gallery': False
    }
    
    # Adaptive update
    if accepted and update_gallery:
        print(f"Claim accepted! Updating gallery with new sample for class {claimed_class_id}...")
        add_probe_to_gallery(image_path, claimed_class_id, gallery_path, model=model)
        result['updated_gallery'] = True
        
    return result

def evaluate_verification(test_path=TEST_PATH, gallery_path=None, metric='cosine'):
    """
    Evaluate the verification system by generating genuine and impostor scores.
    
    Args:
        test_path (str): Path to test probes
        gallery_path (str): Path to gallery file
        metric (str): Distance metric
        
    Returns:
        dict: Evaluation metrics (EER, threshold, scores)
    """
    print("="*60)
    print("VERIFICATION (1:1) EVALUATION")
    print("="*60)
    
    if gallery_path is None:
        gallery_path = GALLERY_PATH
    
    # Load gallery
    print(f"Loading gallery from {gallery_path}...")
    gallery = load_gallery(gallery_path)
    known_classes = set(np.unique(gallery['labels']))
    print(f"Gallery has {len(known_classes)} classes")
    
    # Load model
    print("Loading FaceNet model...")
    model = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Collect probes
    test_probes = [] # (emb, class_id)
    print("Processing test probes...")
    
    # We iterate through all test folders
    # Note: open-set/closed-set doesn't strictly apply to 1:1 the same way,
    # but we can only make genuine claims for classes that exist in the gallery.
    # Impostor claims can be made by anyone against anyone in the gallery.
    
    valid_dirs = []
    if os.path.exists(test_path):
        for d in os.listdir(test_path):
            if os.path.isdir(os.path.join(test_path, d)) and d.isdigit():
                valid_dirs.append(int(d))
    valid_dirs.sort()
            
    total_images = 0
    for class_id in tqdm(valid_dirs, desc="Loading Probes"):
        class_dir = os.path.join(test_path, str(class_id))
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for f in files:
            img_path = os.path.join(class_dir, f)
            # No degradation for standard eval unless requested
            face_img = crop_face(img_path)
            if face_img is not None:
                emb = face_to_embedding(face_img, model=model)
                test_probes.append({'emb': emb, 'id': class_id})
                total_images += 1
                
    print(f"Processed {len(test_probes)} probes.")
    
    genuine_scores = []
    impostor_scores = []
    
    print("Computing verification scores...")
    
    # For every probe, generate claims
    for probe in tqdm(test_probes, desc="Scoring Claims"):
        probe_id = probe['id']
        probe_emb = probe['emb']
        
        # 1. Genuine Claim: Claim to be probe_id
        # Only possible if probe_id is in the gallery
        if probe_id in known_classes:
            score = compute_claim_score(probe_emb, probe_id, gallery, metric)
            if score != float('inf'):
                genuine_scores.append(score)
        
        # 2. Impostor Claims: Claim to be other classes in the gallery
        # We can simulate all other classes or a subset. 
        # For thoroughness, let's do all other known classes.
        for claimed_id in known_classes:
            if claimed_id != probe_id:
                score = compute_claim_score(probe_emb, claimed_id, gallery, metric)
                if score != float('inf'):
                    impostor_scores.append(score)
    
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    print(f"\nGenerated {len(genuine_scores)} genuine scores")
    print(f"Generated {len(impostor_scores)} impostor scores")
    
    # Calculate EER and Optimization
    # Range of thresholds from min to max distance observed
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    t_min, t_max = np.min(all_scores), np.max(all_scores)
    thresholds = np.linspace(t_min, t_max, 200)
    
    far_list = []
    frr_list = []
    
    for t in thresholds:
        # FAR: Impostor scores <= t (Accepted) / Total Impostor
        far = np.mean(impostor_scores <= t)
        # FRR: Genuine scores > t (Rejected) / Total Genuine
        frr = np.mean(genuine_scores > t)
        
        far_list.append(far)
        frr_list.append(frr)
        
    far_list = np.array(far_list)
    frr_list = np.array(frr_list)
    
    # Finding EER where FAR ~= FRR
    # minimize |FAR - FRR|
    eer_idx = np.argmin(np.abs(far_list - frr_list))
    eer = (far_list[eer_idx] + frr_list[eer_idx]) / 2
    best_threshold = thresholds[eer_idx]
    
    print(f"\n{'='*60}")
    print("VERIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Equal Error Rate (EER): {eer:.4f} ({eer*100:.2f}%)")
    print(f"Optimal Threshold (at EER): {best_threshold:.4f}")
    print(f"{'='*60}\n")
    
    return {
        'genuine_scores': genuine_scores,
        'impostor_scores': impostor_scores,
        'thresholds': thresholds,
        'far': far_list,
        'frr': frr_list,
        'eer': eer,
        'best_threshold': best_threshold
    }

def plot_verification_curves(metrics, save_prefix='results/verification'):
    """Plot ROC and DET curves."""
    far = metrics['far']
    frr = metrics['frr']
    tar = 1 - frr # True Acceptance Rate
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    
    # 1. ROC Curve (TAR vs FAR)
    plt.figure(figsize=(8, 8))
    plt.plot(far, tar, label=f'ROC (EER={metrics["eer"]:.2%})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('True Acceptance Rate (TAR)')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_prefix}_roc.png', dpi=300)
    print(f"ROC curve saved to {save_prefix}_roc.png")
    plt.show() # Display plot
    
    # 2. DET Curve (FRR vs FAR) - Log-log scale often used
    plt.figure(figsize=(8, 8))
    plt.plot(far, frr, label=f'DET (EER={metrics["eer"]:.2%})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.title('DET Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'{save_prefix}_det.png', dpi=300)
    print(f"DET curve saved to {save_prefix}_det.png")
    plt.show() # Display plot
    
    # 3. FAR and FRR vs Threshold
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['thresholds'], far, label='FAR', color='red')
    plt.plot(metrics['thresholds'], frr, label='FRR', color='blue')
    plt.axvline(x=metrics['best_threshold'], color='green', linestyle='--', 
                label=f'EER Threshold ({metrics["best_threshold"]:.3f})')
    plt.scatter(metrics['best_threshold'], metrics['eer'], color='green', zorder=5)
    plt.xlabel('Distance Threshold')
    plt.ylabel('Error Rate')
    plt.title('FAR and FRR vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_prefix}_errors.png', dpi=300)
    print(f"Error curves saved to {save_prefix}_errors.png")
    plt.show() # Display plot

if __name__ == "__main__":
    # Run evaluation
    metrics = evaluate_verification(metric='cosine')
    plot_verification_curves(metrics)
    
    # Example operational usage
    print("\n--- Operational Example ---")
    # Simulate a claim
    try:
        # Get first available class from test directory
        available_classes = [d for d in os.listdir(TEST_PATH) 
                            if os.path.isdir(os.path.join(TEST_PATH, d)) and d.isdigit()]
        
        if available_classes:
            first_class_id = int(available_classes[0])
            class_dir = os.path.join(TEST_PATH, str(first_class_id))
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if image_files:
                img_path = os.path.join(class_dir, image_files[0])
                
                # Use the optimized threshold
                optimal_t = metrics['best_threshold']
                
                # Genuine claim
                print(f"Testing GENUINE claim with {img_path} (Claiming ID {first_class_id})")
                res_gen = verify_claim(img_path, first_class_id, optimal_t)
                print(f"Result: {res_gen}")
                
                # Impostor claim (claiming to be a different ID if possible, otherwise use +1)
                impostor_id = first_class_id + 1
                if len(available_classes) > 1:
                    impostor_id = int(available_classes[1])
                
                print(f"Testing IMPOSTOR claim with {img_path} (Claiming ID {impostor_id})")
                res_imp = verify_claim(img_path, impostor_id, optimal_t)
                print(f"Result: {res_imp}")
        else:
            print("No test classes found for operational example.")
    except Exception as e:
        print(f"Could not run example: {e}")
