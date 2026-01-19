import cv2
import numpy as np
from typing import Dict, List


def detect_blinks(frames: List[np.ndarray], face_cascade, eye_cascade) -> Dict[str, any]:
        """Detect eye blinks - strong indicator of liveness"""
        blink_count = 0
        eye_aspect_ratios = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                if len(eyes) >= 2:
                    # Simple eye aspect ratio
                    eye_heights = [eh for (ex, ey, ew, eh) in eyes]
                    ear = np.mean(eye_heights) if eye_heights else 0
                    eye_aspect_ratios.append(ear)
        
        # Detect blinks as sudden drops in eye aspect ratio
        if len(eye_aspect_ratios) > 5:
            for i in range(1, len(eye_aspect_ratios)):
                if eye_aspect_ratios[i] < eye_aspect_ratios[i-1] * 0.7:
                    blink_count += 1
        
        return {
            'blink_count': blink_count,
            'has_blinks': blink_count > 0,
            'eye_movement_variance': float(np.var(eye_aspect_ratios)) if eye_aspect_ratios else 0.0
        }