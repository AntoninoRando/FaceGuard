import cv2
import numpy as np
from typing import Dict


def analyze_color_distribution(face_roi: np.ndarray) -> Dict[str, float]:
    """Analyze color distribution - prints/screens have different color characteristics"""
    # Convert to different color spaces
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    
    # Real skin has specific color distribution
    # Prints/screens have uniform, less natural distribution
    hsv_std = np.std(hsv, axis=(0, 1))
    lab_std = np.std(lab, axis=(0, 1))
    
    # Color diversity score
    color_diversity = float(np.mean(hsv_std) + np.mean(lab_std))
    
    return {
        'color_diversity': color_diversity,
        'hsv_saturation_std': float(hsv_std[1]),
        'lab_a_std': float(lab_std[1])
    }