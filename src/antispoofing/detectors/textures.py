import cv2
import numpy as np
from typing import Dict

from antispoofing.utils.compute_lpb import compute_lbp


def analyze_texture(face_roi: np.ndarray) -> Dict[str, float]:
        """Analyze texture patterns - real faces have different texture than photos/screens"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        
        # Compute LBP
        lbp = compute_lbp(gray)
        
        # Calculate histogram
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256), density=True)
        
        # Real faces have more uniform texture distribution
        # Photos/screens have distinct patterns
        texture_variance = np.var(hist)
        texture_entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        # High-frequency analysis
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Analyze high-frequency content (edges in screens/prints are sharper)
        high_freq_region = magnitude_spectrum[0:center_h//2, :] 
        high_freq_power = np.sum(high_freq_region) / np.sum(magnitude_spectrum)
        
        return {
            'texture_variance': float(texture_variance),
            'texture_entropy': float(texture_entropy),
            'high_freq_ratio': float(high_freq_power),
            'lbp_uniformity': float(np.std(hist))
        }