import cv2
import numpy as np

def detect_moire_patterns(frame: np.ndarray) -> float:
        """Detect moiré patterns common in screen replays"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Fourier transform
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Moiré patterns create periodic artifacts in frequency domain
        h, w = magnitude_spectrum.shape
        center_region = magnitude_spectrum[h//4:3*h//4, w//4:3*w//4]
        
        # Look for periodic peaks (moiré indicators)
        peaks = np.sum(center_region > np.percentile(center_region, 95))
        moire_score = peaks / center_region.size
        
        return float(moire_score)