import cv2
import numpy as np
from typing import Dict, List

from antispoofing.utils.compute_lpb import compute_lbp

def detect_motion_patterns(frames: List[np.ndarray]) -> Dict[str, float]:
        """Analyze motion patterns using optical flow"""
        if len(frames) < 2:
            return {'optical_flow_score': 0.0, 'motion_consistency': 0.0}
        
        flow_magnitudes = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(np.mean(magnitude))
            
            prev_gray = curr_gray
        
        # Real faces have natural, varied motion
        # Videos/photos have rigid, consistent motion
        motion_variance = np.var(flow_magnitudes) if flow_magnitudes else 0.0
        motion_mean = np.mean(flow_magnitudes) if flow_magnitudes else 0.0
        
        return {
            'optical_flow_score': float(motion_mean),
            'motion_consistency': float(motion_variance)
        }