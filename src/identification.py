"""
Face Identification System (1:N)
Supports both closed-set and open-set identification protocols.
"""

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean
import pandas as pd

from sample_utils import crop_face, face_to_embedding, load_gallery, add_degradation
from facenet_pytorch import InceptionResnetV1
from config import TEST_PATH, IDENTIFICATION_THRESHOLD, GALLERY_PATH


def compute_distances(probe_embedding, gallery_embeddings, metric='cosine'):
    """
    Compute distances between probe embedding and all gallery embeddings.
    
    Args:
        probe_embedding (numpy.ndarray): Probe embedding vector (512,)
        gallery_embeddings (numpy.ndarray): Gallery embeddings array (N, 512)
        metric (str): Distance metric ('cosine' or 'euclidean')
    
    Returns:
        numpy.ndarray: Array of distances (N,)
    """
    distances = np.zeros(len(gallery_embeddings))
    
    if metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        for i, gallery_emb in enumerate(gallery_embeddings):
            distances[i] = cosine(probe_embedding, gallery_emb)
    elif metric == 'euclidean':
        for i, gallery_emb in enumerate(gallery_embeddings):
            distances[i] = euclidean(probe_embedding, gallery_emb)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'")
    
    return distances

def identify_probe_closed_set(probe_embedding, gallery, metric='cosine', top_k=10):
    """
    Perform closed-set identification (probe is guaranteed to be in gallery).
    
    Args:
        probe_embedding (numpy.ndarray): Probe embedding vector
        gallery (dict): Gallery data with 'embeddings' and 'labels'
        metric (str): Distance metric ('cosine' or 'euclidean')
        top_k (int): Number of top matches to return
    
    Returns:
        dict: {
            'predicted_label': int,
            'distances': numpy.ndarray,
            'ranked_labels': numpy.ndarray,
            'ranked_indices': numpy.ndarray
        }
    """
    # Compute distances to all gallery embeddings
    distances = compute_distances(probe_embedding, gallery['embeddings'], metric)
    
    # Rank by increasing distance (closest first)
    ranked_indices = np.argsort(distances)
    ranked_labels = gallery['labels'][ranked_indices]
    ranked_distances = distances[ranked_indices]
    
    return {
        'predicted_label': ranked_labels[0],
        'distances': ranked_distances[:top_k],
        'ranked_labels': ranked_labels[:top_k],
        'ranked_indices': ranked_indices[:top_k]
    }

def identify_probe_open_set(probe_embedding, gallery, threshold, metric='cosine', top_k=10):
    """
    Perform open-set identification (probe may not be in gallery).
    
    Args:
        probe_embedding (numpy.ndarray): Probe embedding vector
        gallery (dict): Gallery data with 'embeddings' and 'labels'
        threshold (float): Distance threshold for rejection
        metric (str): Distance metric ('cosine' or 'euclidean')
        top_k (int): Number of top matches to return
    
    Returns:
        dict: {
            'predicted_label': int or -1 (if rejected as unknown),
            'min_distance': float,
            'rejected': bool,
            'distances': numpy.ndarray,
            'ranked_labels': numpy.ndarray
        }
    """
    result = identify_probe_closed_set(probe_embedding, gallery, metric, top_k)
    
    min_distance = result['distances'][0]
    rejected = min_distance > threshold
    
    return {
        'predicted_label': -1 if rejected else result['predicted_label'],
        'min_distance': min_distance,
        'rejected': rejected,
        'distances': result['distances'],
        'ranked_labels': result['ranked_labels']
    }

def process_probe_image(image_path, model, quality_factor=None, noise_factor=None):
    """
    Process a probe image: crop face, optionally degrade, and extract embedding.
    
    Args:
        image_path (str): Path to probe image
        model (InceptionResnetV1): FaceNet model
        quality_factor (float, optional): Quality degradation factor
        noise_factor (float, optional): Noise factor
    
    Returns:
        numpy.ndarray or None: Embedding vector, or None if face not detected
    """
    # Crop face
    face_img = crop_face(image_path, target_size=(160, 160))
    
    if face_img is None:
        return None
    
    # Optionally apply degradation
    if quality_factor is not None or noise_factor is not None:
        qf = quality_factor if quality_factor is not None else 1.0
        nf = noise_factor if noise_factor is not None else 0.0
        face_img = add_degradation(face_img, qf, nf)
    
    # Extract embedding
    embedding = face_to_embedding(face_img, model=model)
    
    return embedding

def evaluate_closed_set(test_path=TEST_PATH, gallery_path=None, 
                       metric='cosine', max_rank=20, apply_degradation=False):
    """
    Evaluate closed-set identification on all test probes.
    Computes CMC curve, Top-k accuracies, and MRR.
    
    Args:
        test_path (str): Path to test probes directory
        gallery_path (str): Path to gallery file
        metric (str): Distance metric
        max_rank (int): Maximum rank for CMC curve
        apply_degradation (bool): Whether to apply random degradation to probes
    
    Returns:
        dict: Evaluation results with CMC, accuracies, MRR, and per-probe results
    """
    print("="*60)
    print("CLOSED-SET IDENTIFICATION EVALUATION")
    print("="*60)
    
    if gallery_path is None:
        gallery_path = GALLERY_PATH
    
    # Load gallery
    print(f"\nLoading gallery from {gallery_path}...")
    gallery = load_gallery(gallery_path)
    print(f"Gallery loaded: {len(gallery['embeddings'])} embeddings, {len(np.unique(gallery['labels']))} classes")
    
    # Load FaceNet model
    print("Loading FaceNet model...")
    model = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Collect all test images
    print(f"\nScanning test probes from {test_path}...")
    test_probes = []
    
    # Dynamically find all class folders in test path
    if os.path.exists(test_path):
        class_folders = [d for d in os.listdir(test_path) 
                        if os.path.isdir(os.path.join(test_path, d)) and d.isdigit()]
        test_class_ids = sorted([int(d) for d in class_folders])
    else:
        test_class_ids = []
        print(f"Warning: Test path {test_path} not found.")

    for class_id in test_class_ids:
        class_folder = os.path.join(test_path, str(class_id))
        
        image_files = [f for f in os.listdir(class_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_folder, img_file)
            test_probes.append((img_path, class_id))
    
    print(f"Found {len(test_probes)} test probes")
    
    # Evaluate each probe
    results = []
    ranks = []  # Rank where correct identity appears
    failed_probes = []
    
    print("\nProcessing probes...")
    for img_path, true_label in tqdm(test_probes):
        try:
            # Process probe
            if apply_degradation:
                qf = np.random.uniform(0.6, 1.0)
                nf = np.random.uniform(0.0, 0.04)
                probe_emb = process_probe_image(img_path, model, qf, nf)
            else:
                probe_emb = process_probe_image(img_path, model)
            
            if probe_emb is None:
                failed_probes.append(img_path)
                continue
            
            # Identify
            result = identify_probe_closed_set(probe_emb, gallery, metric, top_k=max_rank)
            
            # Find rank of correct identity
            correct_rank = None
            for rank, label in enumerate(result['ranked_labels'], 1):
                if label == true_label:
                    correct_rank = rank
                    break
            
            if correct_rank is None:
                correct_rank = max_rank + 1  # Not in top-k
            
            ranks.append(correct_rank)
            results.append({
                'image_path': img_path,
                'true_label': true_label,
                'predicted_label': result['predicted_label'],
                'rank': correct_rank,
                'min_distance': result['distances'][0]
            })
            
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
            failed_probes.append(img_path)
    
    # Compute CMC curve
    ranks = np.array(ranks)
    cmc = np.zeros(max_rank)
    
    for k in range(1, max_rank + 1):
        cmc[k-1] = np.sum(ranks <= k) / len(ranks)
    
    # Compute MRR (Mean Reciprocal Rank)
    reciprocal_ranks = [1.0 / r if r <= max_rank else 0.0 for r in ranks]
    mrr = np.mean(reciprocal_ranks)
    
    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total probes: {len(test_probes)}")
    print(f"Successfully processed: {len(results)}")
    print(f"Failed probes: {len(failed_probes)}")
    print(f"\nMetric: {metric}")
    print(f"Degradation applied: {apply_degradation}")
    print(f"\nCMC Accuracies:")
    print(f"  Top-1 (Rank-1):  {cmc[0]:.4f} ({cmc[0]*100:.2f}%)")
    print(f"  Top-5 (Rank-5):  {cmc[4]:.4f} ({cmc[4]*100:.2f}%)")
    print(f"  Top-10 (Rank-10): {cmc[9]:.4f} ({cmc[9]*100:.2f}%)")
    print(f"\nMean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"{'='*60}\n")
    
    return {
        'cmc': cmc,
        'ranks': ranks,
        'mrr': mrr,
        'results': results,
        'failed_probes': failed_probes,
        'metric': metric,
        'top1': cmc[0],
        'top5': cmc[4],
        'top10': cmc[9]
    }

def evaluate_open_set(test_path=TEST_PATH, gallery_path=None,
                     threshold=None, metric='cosine', include_unknown=True):
    """
    Evaluate open-set identification with threshold-based rejection.
    
    Args:
        test_path (str): Path to test probes directory
        gallery_path (str): Path to gallery file
        threshold (float): Distance threshold for rejection
        metric (str): Distance metric
        include_unknown (bool): If True, treats classes 31-32 as unknown (not in gallery)
    
    Returns:
        dict: Evaluation results with accuracy, precision, recall, etc.
    """
    print("="*60)
    print("OPEN-SET IDENTIFICATION EVALUATION")
    print("="*60)
    
    # Use default threshold from config if not provided
    if threshold is None:
        threshold = IDENTIFICATION_THRESHOLD
        print(f"Using default threshold from config: {threshold}")
    
    if gallery_path is None:
        gallery_path = GALLERY_PATH
    
    # Load gallery
    print(f"\nLoading gallery from {gallery_path}...")
    gallery = load_gallery(gallery_path)
    known_classes = set(gallery['labels'])
    print(f"Gallery loaded: {len(gallery['embeddings'])} embeddings")
    print(f"Known classes in gallery: {sorted(known_classes)}")
    
    # Load FaceNet model
    print("Loading FaceNet model...")
    model = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Collect test probes
    print(f"\nScanning test probes from {test_path}...")
    test_probes = []
    
    # Dynamically find all class folders in test path
    if os.path.exists(test_path):
        class_folders = [d for d in os.listdir(test_path) 
                        if os.path.isdir(os.path.join(test_path, d)) and d.isdigit()]
        test_class_ids = sorted([int(d) for d in class_folders])
    else:
        test_class_ids = []
        print(f"Warning: Test path {test_path} not found.")

    for class_id in test_class_ids:
        class_folder = os.path.join(test_path, str(class_id))
        
        # Determine if this is a known or unknown class
        is_known = class_id in known_classes
        
        image_files = [f for f in os.listdir(class_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_folder, img_file)
            test_probes.append((img_path, class_id, is_known))
    
    print(f"Found {len(test_probes)} test probes")
    
    # Evaluate
    results = []
    tp_known = 0  # True positive: known correctly identified
    fp_known = 0  # False positive: unknown classified as known
    tn_unknown = 0  # True negative: unknown correctly rejected
    fn_unknown = 0  # False negative: known incorrectly rejected
    
    print(f"\nProcessing probes with threshold={threshold}...")
    for img_path, true_label, is_known in tqdm(test_probes):
        try:
            probe_emb = process_probe_image(img_path, model)
            
            if probe_emb is None:
                continue
            
            # Identify with threshold
            result = identify_probe_open_set(probe_emb, gallery, threshold, metric)
            
            predicted_label = result['predicted_label']
            rejected = result['rejected']
            
            # Update confusion matrix
            if is_known:
                if not rejected and predicted_label == true_label:
                    tp_known += 1
                elif rejected:
                    fn_unknown += 1
                # else: misclassified (counted separately)
            else:
                if rejected:
                    tn_unknown += 1
                else:
                    fp_known += 1
            
            results.append({
                'image_path': img_path,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'is_known': is_known,
                'rejected': rejected,
                'min_distance': result['min_distance']
            })
            
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
    
    # Compute metrics
    total_known = sum(1 for r in results if r['is_known'])
    total_unknown = sum(1 for r in results if not r['is_known'])
    
    accuracy_known = tp_known / total_known if total_known > 0 else 0
    rejection_rate_unknown = tn_unknown / total_unknown if total_unknown > 0 else 0
    
    # Detection metrics
    detection_accuracy = (tp_known + tn_unknown) / len(results) if len(results) > 0 else 0
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Threshold: {threshold}")
    print(f"Metric: {metric}")
    print(f"\nDataset:")
    print(f"  Known probes: {total_known}")
    print(f"  Unknown probes: {total_unknown}")
    print(f"\nPerformance:")
    print(f"  Known identification accuracy: {accuracy_known:.4f} ({accuracy_known*100:.2f}%)")
    print(f"  Unknown rejection rate: {rejection_rate_unknown:.4f} ({rejection_rate_unknown*100:.2f}%)")
    print(f"  Overall detection accuracy: {detection_accuracy:.4f} ({detection_accuracy*100:.2f}%)")
    print(f"\nConfusion:")
    print(f"  TP (known correctly identified): {tp_known}")
    print(f"  FP (unknown as known): {fp_known}")
    print(f"  TN (unknown correctly rejected): {tn_unknown}")
    print(f"  FN (known incorrectly rejected): {fn_unknown}")
    print(f"{'='*60}\n")
    
    return {
        'threshold': threshold,
        'metric': metric,
        'results': results,
        'accuracy_known': accuracy_known,
        'rejection_rate_unknown': rejection_rate_unknown,
        'detection_accuracy': detection_accuracy,
        'tp': tp_known,
        'fp': fp_known,
        'tn': tn_unknown,
        'fn': fn_unknown
    }

def plot_cmc_curve(evaluation_results, save_path='results/cmc_curve.png'):
    """
    Plot CMC curve from evaluation results.
    
    Args:
        evaluation_results (dict): Results from evaluate_closed_set
        save_path (str): Path to save the plot
    """
    cmc = evaluation_results['cmc']
    metric = evaluation_results['metric']
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cmc) + 1), cmc, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Recognition Rate', fontsize=12)
    plt.title(f'CMC Curve - Closed-Set Identification ({metric} distance)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, len(cmc))
    plt.ylim(0, 1.05)
    
    # Add top-k annotations
    plt.axhline(y=cmc[0], color='r', linestyle='--', alpha=0.5, label=f'Top-1: {cmc[0]:.3f}')
    plt.axhline(y=cmc[4], color='g', linestyle='--', alpha=0.5, label=f'Top-5: {cmc[4]:.3f}')
    plt.axhline(y=cmc[9], color='orange', linestyle='--', alpha=0.5, label=f'Top-10: {cmc[9]:.3f}')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"CMC curve saved to {save_path}")
    plt.show()

def find_optimal_threshold(test_path=TEST_PATH, gallery_path=None,
                          metric='cosine', thresholds=None):
    """
    Find optimal threshold for open-set identification by testing multiple values.
    
    Args:
        test_path (str): Path to test probes
        gallery_path (str): Path to gallery
        metric (str): Distance metric
        thresholds (list): List of thresholds to test
    
    Returns:
        dict: Results for each threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.2, 0.8, 13)
    
    if gallery_path is None:
        gallery_path = GALLERY_PATH
    
    print("Finding optimal threshold...")
    print(f"Testing {len(thresholds)} threshold values: {thresholds}")
    
    results = []
    
    for threshold in thresholds:
        print(f"\n--- Testing threshold={threshold:.3f} ---")
        eval_result = evaluate_open_set(test_path, gallery_path, threshold, metric, include_unknown=True)
        
        results.append({
            'threshold': threshold,
            'detection_accuracy': eval_result['detection_accuracy'],
            'accuracy_known': eval_result['accuracy_known'],
            'rejection_rate_unknown': eval_result['rejection_rate_unknown']
        })
    
    # Find best threshold
    best_idx = np.argmax([r['detection_accuracy'] for r in results])
    best_result = results[best_idx]
    
    print(f"\n{'='*60}")
    print("OPTIMAL THRESHOLD SEARCH RESULTS")
    print(f"{'='*60}")
    print(f"Best threshold: {best_result['threshold']:.3f}")
    print(f"Detection accuracy: {best_result['detection_accuracy']:.4f}")
    print(f"Known identification accuracy: {best_result['accuracy_known']:.4f}")
    print(f"Unknown rejection rate: {best_result['rejection_rate_unknown']:.4f}")
    print(f"{'='*60}\n")
    
    return {
        'all_results': results,
        'best_threshold': best_result['threshold'],
        'best_result': best_result
    }

if __name__ == "__main__":
    # Example usage
    print("Face Identification Evaluation System\n")
    
    # Closed-set evaluation
    print("Running closed-set evaluation...")
    closed_results = evaluate_closed_set(
        test_path=TEST_PATH,
        gallery_path=GALLERY_PATH,
        metric='cosine',
        max_rank=20,
        apply_degradation=False
    )
    
    # Plot CMC curve
    plot_cmc_curve(closed_results, save_path='results/cmc_curve.png')
    
    # Open-set evaluation
    print("\nRunning open-set evaluation...")
    open_results = evaluate_open_set(
        test_path=TEST_PATH,
        gallery_path=GALLERY_PATH,
        threshold=IDENTIFICATION_THRESHOLD,
        metric='cosine'
    )
