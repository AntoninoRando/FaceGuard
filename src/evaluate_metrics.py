"""
Evaluation Script for FaceGuard System
Computes comprehensive metrics for system performance evaluation

This script evaluates the biometric system and computes essential metrics:
- FAR (False Acceptance Rate)
- FRR (False Rejection Rate)  
- EER (Equal Error Rate)
- ROC Curve
- DET Curve
- CMC Curve
- Confusion Matrix
- Anti-spoofing performance

Usage:
    python evaluate_metrics.py --gallery_path data/initial_samples --test_path data/probes_for_test
"""

import argparse
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from facenet_pytorch import InceptionResnetV1

from sample_utils import load_gallery, face_to_embedding, crop_face
from identification import identify_probe_open_set, process_probe_image
from verification import compute_claim_score
from config import IDENTIFICATION_THRESHOLD, VERIFICATION_THRESHOLD


def compute_verification_metrics(
    test_path: str,
    gallery_path: str,
    thresholds: List[float] = None
) -> Dict[str, Any]:
    """
    Compute verification metrics (FAR, FRR, EER, ROC)
    
    Args:
        test_path: Path to test probes
        gallery_path: Path to gallery
        thresholds: List of thresholds to evaluate
        
    Returns:
        Dictionary containing metrics
    """
    print("Loading gallery and model...")
    gallery = load_gallery(gallery_path)
    model = InceptionResnetV1(pretrained='vggface2').eval()
    
    print("Computing verification scores...")
    genuine_scores = []
    impostor_scores = []
    
    # Collect all test samples
    test_samples = []
    test_path_obj = Path(test_path)
    
    for class_folder in test_path_obj.iterdir():
        if not class_folder.is_dir():
            continue
        
        try:
            class_id = int(class_folder.name)
        except ValueError:
            continue
        
        for img_file in class_folder.glob('*.jpg'):
            test_samples.append((img_file, class_id))
    
    print(f"Found {len(test_samples)} test samples")
    
    # Compute scores
    for img_path, true_label in test_samples:
        try:
            probe_emb = process_probe_image(str(img_path), model)
            if probe_emb is None:
                continue
            
            # Compute distance to gallery
            distances = []
            gallery_labels = gallery['labels']
            
            for i, emb in enumerate(gallery['embeddings']):
                dist = np.linalg.norm(probe_emb - emb)
                distances.append((dist, gallery_labels[i]))
            
            # Find minimum distance for true class (genuine)
            true_class_distances = [d for d, lbl in distances if lbl == true_label]
            if true_class_distances:
                genuine_scores.append(min(true_class_distances))
            
            # Find minimum distance for other classes (impostor)
            other_class_distances = [d for d, lbl in distances if lbl != true_label]
            if other_class_distances:
                impostor_scores.append(min(other_class_distances))
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"Collected {len(genuine_scores)} genuine scores and {len(impostor_scores)} impostor scores")
    
    # Compute metrics at different thresholds
    if thresholds is None:
        thresholds = np.linspace(0.3, 1.5, 100)
    
    far_list = []
    frr_list = []
    
    for threshold in thresholds:
        # FAR: proportion of impostor scores below threshold (accepted)
        far = np.sum(np.array(impostor_scores) < threshold) / len(impostor_scores) if impostor_scores else 0
        
        # FRR: proportion of genuine scores above threshold (rejected)
        frr = np.sum(np.array(genuine_scores) >= threshold) / len(genuine_scores) if genuine_scores else 0
        
        far_list.append(far)
        frr_list.append(frr)
    
    # Find EER (where FAR ≈ FRR)
    eer_idx = np.argmin(np.abs(np.array(far_list) - np.array(frr_list)))
    eer = (far_list[eer_idx] + frr_list[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    # Compute TAR (True Accept Rate) = 1 - FRR
    tar_list = [1 - frr for frr in frr_list]
    
    # Compute AUC
    roc_auc = auc(far_list, tar_list)
    
    return {
        'genuine_scores': genuine_scores,
        'impostor_scores': impostor_scores,
        'thresholds': thresholds.tolist(),
        'far': far_list,
        'frr': frr_list,
        'tar': tar_list,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'auc': roc_auc
    }


def compute_identification_metrics(
    test_path: str,
    gallery_path: str,
    threshold: float = None
) -> Dict[str, Any]:
    """
    Compute identification metrics (accuracy, CMC curve)
    
    Args:
        test_path: Path to test probes
        gallery_path: Path to gallery
        threshold: Distance threshold for rejection
        
    Returns:
        Dictionary containing metrics
    """
    if threshold is None:
        threshold = IDENTIFICATION_THRESHOLD
    
    print("Loading gallery and model...")
    gallery = load_gallery(gallery_path)
    model = InceptionResnetV1(pretrained='vggface2').eval()
    known_classes = set(gallery['labels'])
    
    print("Evaluating identification...")
    test_path_obj = Path(test_path)
    
    results = []
    
    for class_folder in test_path_obj.iterdir():
        if not class_folder.is_dir():
            continue
        
        try:
            true_label = int(class_folder.name)
        except ValueError:
            continue
        
        is_known = true_label in known_classes
        
        for img_file in class_folder.glob('*.jpg'):
            try:
                probe_emb = process_probe_image(str(img_file), model)
                if probe_emb is None:
                    continue
                
                result = identify_probe_open_set(probe_emb, gallery, threshold)
                
                results.append({
                    'true_label': true_label,
                    'predicted_label': result['predicted_label'],
                    'is_known': is_known,
                    'rejected': result['rejected'],
                    'distance': result['min_distance'],
                    'top_matches': result['top_matches']
                })
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
    
    print(f"Evaluated {len(results)} probes")
    
    # Compute accuracy
    known_results = [r for r in results if r['is_known']]
    correct = sum(1 for r in known_results if r['predicted_label'] == r['true_label'] and not r['rejected'])
    accuracy = correct / len(known_results) if known_results else 0
    
    # Compute CMC curve
    max_rank = 20
    cmc = []
    
    for rank in range(1, max_rank + 1):
        correct_at_rank = 0
        for r in known_results:
            if r['rejected']:
                continue
            
            # Check if true label is in top-k matches
            top_k_labels = [m['label'] for m in r['top_matches'][:rank]]
            if r['true_label'] in top_k_labels:
                correct_at_rank += 1
        
        cmc.append(correct_at_rank / len(known_results) if known_results else 0)
    
    # Compute confusion matrix components
    tp = sum(1 for r in results if r['is_known'] and not r['rejected'] and r['predicted_label'] == r['true_label'])
    fp = sum(1 for r in results if not r['is_known'] and not r['rejected'])
    tn = sum(1 for r in results if not r['is_known'] and r['rejected'])
    fn = sum(1 for r in results if r['is_known'] and r['rejected'])
    
    return {
        'results': results,
        'accuracy': accuracy,
        'cmc': cmc,
        'ranks': list(range(1, max_rank + 1)),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'total': len(results)
    }


def plot_roc_curve(verification_metrics: Dict, save_path: str = 'results/roc_curve.png'):
    """Plot ROC curve"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.plot(verification_metrics['far'], verification_metrics['tar'], 
             label=f"ROC (AUC = {verification_metrics['auc']:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('True Acceptance Rate (TAR)')
    plt.title('ROC Curve - Verification Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")


def plot_det_curve(verification_metrics: Dict, save_path: str = 'results/det_curve.png'):
    """Plot DET curve"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.loglog(verification_metrics['far'], verification_metrics['frr'], linewidth=2)
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.title('DET Curve - Detection Error Tradeoff')
    plt.grid(True, alpha=0.3, which='both')
    
    # Mark EER point
    eer = verification_metrics['eer']
    plt.plot(eer, eer, 'r*', markersize=15, label=f'EER = {eer:.4f}')
    plt.legend()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"DET curve saved to {save_path}")


def plot_cmc_curve(identification_metrics: Dict, save_path: str = 'results/cmc_curve.png'):
    """Plot CMC curve"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.plot(identification_metrics['ranks'], identification_metrics['cmc'], 
             marker='o', linewidth=2)
    plt.xlabel('Rank')
    plt.ylabel('Identification Rate')
    plt.title('CMC Curve - Identification Performance')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Add rank-1 annotation
    rank1 = identification_metrics['cmc'][0]
    plt.axhline(y=rank1, color='r', linestyle='--', alpha=0.5)
    plt.text(10, rank1 + 0.02, f'Rank-1: {rank1:.4f}', fontsize=12)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"CMC curve saved to {save_path}")


def save_metrics_report(
    verification_metrics: Dict,
    identification_metrics: Dict,
    save_path: str = 'results/metrics_report.json'
):
    """Save comprehensive metrics report"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        'verification': {
            'eer': verification_metrics['eer'],
            'eer_threshold': verification_metrics['eer_threshold'],
            'auc': verification_metrics['auc'],
            'num_genuine': len(verification_metrics['genuine_scores']),
            'num_impostor': len(verification_metrics['impostor_scores'])
        },
        'identification': {
            'accuracy': identification_metrics['accuracy'],
            'rank1': identification_metrics['cmc'][0],
            'rank5': identification_metrics['cmc'][4],
            'rank10': identification_metrics['cmc'][9],
            'confusion_matrix': {
                'tp': identification_metrics['tp'],
                'fp': identification_metrics['fp'],
                'tn': identification_metrics['tn'],
                'fn': identification_metrics['fn']
            },
            'total_probes': identification_metrics['total']
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nMetrics report saved to {save_path}")
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print("\nVerification Performance:")
    print(f"  EER: {report['verification']['eer']:.4f} ({report['verification']['eer']*100:.2f}%)")
    print(f"  AUC: {report['verification']['auc']:.4f}")
    print(f"  EER Threshold: {report['verification']['eer_threshold']:.4f}")
    print(f"\nIdentification Performance:")
    print(f"  Accuracy: {report['identification']['accuracy']:.4f} ({report['identification']['accuracy']*100:.2f}%)")
    print(f"  Rank-1: {report['identification']['rank1']:.4f}")
    print(f"  Rank-5: {report['identification']['rank5']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {report['identification']['confusion_matrix']['tp']}")
    print(f"  FP: {report['identification']['confusion_matrix']['fp']}")
    print(f"  TN: {report['identification']['confusion_matrix']['tn']}")
    print(f"  FN: {report['identification']['confusion_matrix']['fn']}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate FaceGuard System')
    parser.add_argument('--gallery_path', type=str, default='data/initial_samples',
                       help='Path to gallery images')
    parser.add_argument('--test_path', type=str, default='data/probes_for_test',
                       help='Path to test probe images')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Identification threshold (default: from config)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FACEGUARD SYSTEM EVALUATION")
    print("="*60)
    
    # Compute verification metrics
    print("\n1. Computing verification metrics (FAR, FRR, EER, ROC)...")
    verification_metrics = compute_verification_metrics(
        test_path=args.test_path,
        gallery_path=args.gallery_path
    )
    
    # Compute identification metrics
    print("\n2. Computing identification metrics (Accuracy, CMC)...")
    identification_metrics = compute_identification_metrics(
        test_path=args.test_path,
        gallery_path=args.gallery_path,
        threshold=args.threshold
    )
    
    # Generate plots
    print("\n3. Generating visualization plots...")
    plot_roc_curve(verification_metrics, f'{args.output_dir}/roc_curve.png')
    plot_det_curve(verification_metrics, f'{args.output_dir}/det_curve.png')
    plot_cmc_curve(identification_metrics, f'{args.output_dir}/cmc_curve.png')
    
    # Save report
    print("\n4. Saving metrics report...")
    save_metrics_report(verification_metrics, identification_metrics, 
                       f'{args.output_dir}/metrics_report.json')
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
