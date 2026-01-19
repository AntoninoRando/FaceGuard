"""
Video Anti-Spoofing Application
Detects presentation attacks: printed photos, video replay, and masks
"""

import cv2
import os
from typing import Tuple, Dict
import json
from datetime import datetime

from antispoofing.detectors.color_distribution import analyze_color_distribution
from antispoofing.detectors.eye_blinks import detect_blinks
from antispoofing.detectors.more_patterns import detect_moire_patterns
from antispoofing.detectors.motion_patterns import detect_motion_patterns
from antispoofing.detectors.textures import analyze_texture

class AntiSpoofingDetector:
    """
    Multi-method anti-spoofing detector combining:
    1. Texture analysis (LBP - Local Binary Patterns)
    2. Optical flow analysis (motion patterns)
    3. Frequency domain analysis (moiré patterns)
    4. Liveness detection (eye blinking, subtle movements)
    """
    
    def __init__(self, config_path: str = None):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'antispoofing_config.json')
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.thresholds = self.config['thresholds']
        self.weights = self.config['weights']
        self.max_possible_scores = self.config['max_possible_scores']
        self.classification_config = self.config['classification']
    
    
    def classify_spoof_type(self, features: Dict) -> Tuple[str, float, Dict]:
        """
        Classify the type of attack based on collected features using weighted scoring
        Returns: (classification, confidence, reasoning)
        """
        scores = {
            'real': 0.0,
            'photo': 0.0,
            'video_replay': 0.0,
            'mask': 0.0
        }
        
        reasoning = {}
        weights = {}
        
        # Extract features with defaults
        motion_consistency = features.get('motion_consistency', 0.0)
        has_blinks = features.get('has_blinks', False)
        blink_count = features.get('blink_count', 0)
        moire_score = features.get('moire_score', 0.0)
        texture_variance = features.get('texture_variance', 0.0)
        texture_entropy = features.get('texture_entropy', 0.0)
        high_freq_ratio = features.get('high_freq_ratio', 0.0)
        color_diversity = features.get('color_diversity', 0.0)
        optical_flow = features.get('optical_flow_score', 0.0)
        lbp_uniformity = features.get('lbp_uniformity', 0.0)
        
        # ===== REAL FACE SCORING =====
        # Liveness indicators (strongest evidence)
        if has_blinks and blink_count >= self.thresholds['blink_count']['multiple_blinks_min']:
            weight = self.weights['real']['strong_blinks']
            scores['real'] += weight
            weights['strong_blinks'] = weight
            reasoning['strong_blinks'] = f'Multiple blinks detected ({blink_count})'
        elif has_blinks:
            weight = self.weights['real']['blinks']
            scores['real'] += weight
            weights['blinks'] = weight
            reasoning['blinks'] = 'Eye blinks detected'
        
        # Natural motion patterns
        if motion_consistency > self.thresholds['motion_consistency']['strong']:
            weight = self.weights['real']['strong_motion']
            scores['real'] += weight
            weights['strong_motion'] = weight
            reasoning['strong_motion'] = 'Strong natural motion detected'
        elif motion_consistency > self.thresholds['motion_consistency']['moderate']:
            weight = self.weights['real']['moderate_motion']
            scores['real'] += weight
            weights['moderate_motion'] = weight
            reasoning['moderate_motion'] = 'Natural motion present'
        elif motion_consistency > self.thresholds['motion_consistency']['slight']:
            weight = self.weights['real']['slight_motion']
            scores['real'] += weight
            weights['slight_motion'] = weight
            reasoning['slight_motion'] = 'Slight motion detected'
        
        # Optical flow indicates actual movement
        if optical_flow > self.thresholds['optical_flow']['continuous_motion_min']:
            weight = self.weights['real']['optical_flow']
            scores['real'] += weight
            weights['optical_flow'] = weight
            reasoning['optical_flow'] = 'Continuous motion detected'
        
        # Natural texture characteristics
        if self.thresholds['texture_variance']['natural_min'] < texture_variance < self.thresholds['texture_variance']['natural_max']:
            weight = self.weights['real']['natural_texture']
            scores['real'] += weight
            weights['natural_texture'] = weight
            reasoning['natural_texture'] = 'Natural skin texture variance'
        
        if texture_entropy > self.thresholds['texture_entropy']['natural_min']:
            weight = self.weights['real']['texture_complexity']
            scores['real'] += weight
            weights['texture_complexity'] = weight
            reasoning['texture_complexity'] = 'Complex texture patterns (natural skin)'
        
        # Natural color distribution
        if color_diversity > self.thresholds['color_diversity']['natural_high_min']:
            weight = self.weights['real']['color_variance_high']
            scores['real'] += weight
            weights['color_variance'] = weight
            reasoning['color_variance'] = 'Natural color distribution'
        elif color_diversity > self.thresholds['color_diversity']['natural_adequate_min']:
            weight = self.weights['real']['color_variance_adequate']
            scores['real'] += weight
            weights['color_variance'] = weight
            reasoning['color_variance'] = 'Adequate color variance'
        
        # Low moiré patterns (no screen)
        if moire_score < self.thresholds['moire_score']['real_max']:
            weight = self.weights['real']['no_moire']
            scores['real'] += weight
            weights['no_moire'] = weight
            reasoning['no_moire'] = 'No screen artifacts'
        
        # ===== PHOTO ATTACK SCORING =====
        # Static image indicators
        if motion_consistency < self.thresholds['motion_consistency']['static_photo_max'] and not has_blinks:
            weight = self.weights['photo']['static_photo']
            scores['photo'] += weight
            weights['static_photo'] = weight
            reasoning['static_photo'] = 'No motion or blinks detected (static image)'
        elif motion_consistency < self.thresholds['motion_consistency']['minimal_motion_max']:
            weight = self.weights['photo']['minimal_motion']
            scores['photo'] += weight
            weights['minimal_motion'] = weight
            reasoning['minimal_motion'] = 'Almost no motion detected'
        
        # Low texture complexity (printed)
        if texture_variance < self.thresholds['texture_variance']['photo_uniform_max']:
            weight = self.weights['photo']['uniform_texture']
            scores['photo'] += weight
            weights['uniform_texture'] = weight
            reasoning['uniform_texture'] = 'Extremely uniform texture (printed photo)'
        elif texture_variance < self.thresholds['texture_variance']['photo_low_max']:
            weight = self.weights['photo']['low_texture']
            scores['photo'] += weight
            weights['low_texture'] = weight
            reasoning['low_texture'] = 'Low texture variance'
        
        # Low color diversity (printed colors)
        if color_diversity < self.thresholds['color_diversity']['photo_poor_max']:
            weight = self.weights['photo']['poor_colors']
            scores['photo'] += weight
            weights['poor_colors'] = weight
            reasoning['poor_colors'] = 'Limited color range (print)'
        
        # Flat texture entropy
        if texture_entropy < self.thresholds['texture_entropy']['photo_flat_max']:
            weight = self.weights['photo']['flat_texture']
            scores['photo'] += weight
            weights['flat_texture'] = weight
            reasoning['flat_texture'] = 'Flat texture characteristics'
        
        # ===== VIDEO REPLAY SCORING =====
        # Moiré patterns (screen artifacts)
        if moire_score > self.thresholds['moire_score']['video_strong_min']:
            weight = self.weights['video_replay']['strong_moire']
            scores['video_replay'] += weight
            weights['strong_moire'] = weight
            reasoning['strong_moire'] = 'Strong moiré patterns (screen replay)'
        elif moire_score > self.thresholds['moire_score']['video_moderate_min']:
            weight = self.weights['video_replay']['moderate_moire']
            scores['video_replay'] += weight
            weights['moderate_moire'] = weight
            reasoning['moderate_moire'] = 'Moiré patterns detected'
        elif moire_score > self.thresholds['moire_score']['video_slight_min']:
            weight = self.weights['video_replay']['slight_moire']
            scores['video_replay'] += weight
            weights['slight_moire'] = weight
            reasoning['slight_moire'] = 'Slight screen artifacts'
        
        # Sharp digital edges
        if high_freq_ratio > self.thresholds['high_freq_ratio']['video_sharp_min']:
            weight = self.weights['video_replay']['sharp_edges']
            scores['video_replay'] += weight
            weights['sharp_edges'] = weight
            reasoning['sharp_edges'] = 'Unnaturally sharp edges (digital)'
        elif high_freq_ratio > self.thresholds['high_freq_ratio']['video_digital_min']:
            weight = self.weights['video_replay']['digital_edges']
            scores['video_replay'] += weight
            weights['digital_edges'] = weight
            reasoning['digital_edges'] = 'Digital edge characteristics'
        
        # Rigid motion (video playback)
        if self.thresholds['motion_consistency']['video_rigid_min'] < motion_consistency < self.thresholds['motion_consistency']['video_rigid_max'] and not has_blinks:
            weight = self.weights['video_replay']['rigid_motion']
            scores['video_replay'] += weight
            weights['rigid_motion'] = weight
            reasoning['rigid_motion'] = 'Rigid motion without natural behavior'
        
        # Blink detection without natural motion
        if has_blinks and optical_flow < 0.9:
            scores['video_replay'] += 15.0
            reasoning['blinks_without_flow'] = 'Blinks detected but insufficient natural motion'
        
        # Detect replay-with-blinks pattern
        if has_blinks and motion_consistency < 0.2 and optical_flow < 1.0:
            scores['video_replay'] += 20.0
            reasoning['replay_with_blinks'] = 'Blinks detected but motion is rigid (replay)'
        
        # ===== MASK ATTACK SCORING =====
        # Artificial material texture
        if lbp_uniformity < self.thresholds['lbp_uniformity']['mask_uniform_max']:
            weight = self.weights['mask']['uniform_surface']
            scores['mask'] += weight
            weights['uniform_surface'] = weight
            reasoning['uniform_surface'] = 'Uniform surface texture (mask material)'
        
        if texture_entropy < self.thresholds['texture_entropy']['mask_material_max'] and color_diversity < self.thresholds['color_diversity']['mask_artificial_max']:
            weight = self.weights['mask']['artificial_material']
            scores['mask'] += weight
            weights['artificial_material'] = weight
            reasoning['artificial_material'] = 'Artificial material characteristics'
        
        # Unnatural but present motion (mask worn by person)
        if motion_consistency > self.thresholds['motion_consistency']['mask_motion_min'] and not has_blinks and texture_variance < self.thresholds['texture_variance']['mask_max']:
            weight = self.weights['mask']['mask_motion']
            scores['mask'] += weight
            weights['mask_motion'] = weight
            reasoning['mask_motion'] = 'Motion without natural facial features'
        
        # ===== FINAL CLASSIFICATION =====
        # Normalize scores to 0-100 range
        normalized_scores = {
            k: (v / self.max_possible_scores[k]) * 100.0 
            for k, v in scores.items()
        }
        
        # Get classification
        classification = max(normalized_scores, key=normalized_scores.get)
        confidence = normalized_scores[classification]
        
        # Apply threshold: need minimum confidence to classify as spoof
        min_spoof_conf = self.classification_config['min_spoof_confidence']
        min_real_fallback = self.classification_config['min_real_fallback_confidence']
        
        if classification != 'real' and confidence < min_spoof_conf:
            classification = 'real'
            confidence = normalized_scores['real'] if normalized_scores['real'] > min_real_fallback else 50
            reasoning['default_to_real'] = 'Insufficient evidence for spoofing'
        
        # If real score is high, prefer it
        prefer_real_thresh = self.classification_config['prefer_real_threshold']
        if normalized_scores['real'] > prefer_real_thresh:
            classification = 'real'
            confidence = normalized_scores['real']
        
        return classification, confidence, reasoning

    def analyze_video(self, video_path: str, max_frames: int = 100) -> Dict:
        """
        Analyze a video file for spoofing detection (for REST API use)
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to analyze
            
        Returns:
            Dict with analysis results
        """
        if not os.path.exists(video_path):
            return {'error': f'Video file not found: {video_path}'}
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'error': 'Could not open video file'}
        
        frames = []
        face_rois = []
        frame_count = 0
        
        # Extract frames
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 2 == 0:  # Sample every 2nd frame
                frames.append(frame)
                
                # Detect face
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face_roi = frame[y:y+h, x:x+w]
                    face_rois.append(face_roi)
        
        cap.release()
        
        if len(frames) == 0:
            return {'error': 'No frames could be extracted from video'}
        
        if len(face_rois) == 0:
            return {'error': 'No faces detected in video'}
        
        # Collect all features
        features = {}
        
        # Texture analysis on middle face (usually better quality)
        mid_idx = len(face_rois) // 2
        texture_features = analyze_texture(face_rois[mid_idx])
        features.update(texture_features)
        
        # Motion analysis on sufficient frames
        frames_for_motion = frames[:min(30, len(frames))]  # Use more frames for better motion analysis
        motion_features = detect_motion_patterns(frames_for_motion)
        features.update(motion_features)
        
        # Blink detection
        blink_features = detect_blinks(frames, self.face_cascade, self.eye_cascade)
        features.update(blink_features)
        
        # Moiré pattern detection - check multiple frames
        moire_scores = []
        sample_indices = [len(frames)//4, len(frames)//2, 3*len(frames)//4]
        for idx in sample_indices:
            if idx < len(frames):
                moire_scores.append(detect_moire_patterns(frames[idx]))
        features['moire_score'] = max(moire_scores) if moire_scores else 0.0
        
        # Color analysis - average over multiple faces
        color_diversities = []
        for face_roi in face_rois[::max(1, len(face_rois)//5)]:  # Sample 5 faces
            color_feats = analyze_color_distribution(face_roi)
            color_diversities.append(color_feats['color_diversity'])
        
        color_features = analyze_color_distribution(face_rois[mid_idx])
        color_features['color_diversity'] = sum(color_diversities) / len(color_diversities)  # Average
        
        # Color analysis
        color_features = analyze_color_distribution(face_rois[0])
        features.update(color_features)
        
        # Classify
        classification, confidence, reasoning = self.classify_spoof_type(features)
        
        # Calculate spoof score (0 = real, 1 = spoofed)
        is_live = classification == 'real'
        spoof_score = 1.0 - (confidence / 100.0) if is_live else (confidence / 100.0)
        
        return {
            'is_live': is_live,
            'spoof_score': spoof_score,
            'confidence': confidence / 100.0,
            'classification': classification,
            'frames_analyzed': len(frames),
            'faces_detected': len(face_rois),
            'features': {k: round(v, 4) if isinstance(v, float) else v 
                        for k, v in features.items()},
            'reasoning': reasoning
        }


def save_results(results: Dict, output_path: str):
    """Save analysis results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

def print_results(results: Dict):
    """Print formatted results"""
    if 'error' in results:
        print(f"\n❌ ERROR: {results['error']}")
        return
    
    print("\n" + "="*60)
    print("ANTI-SPOOFING ANALYSIS RESULTS")
    print("="*60)
    
    classification = results['classification']
    confidence = results.get('confidence', 0) * 100  # Convert to percentage
    
    # Status symbol
    if classification == 'real':
        symbol = "✅"
        status = "REAL FACE (LIVE)"
    elif classification == 'photo':
        symbol = "🚫"
        status = "PHOTO ATTACK"
    elif classification == 'video_replay':
        symbol = "📺"
        status = "VIDEO REPLAY ATTACK"
    elif classification == 'mask':
        symbol = "🎭"
        status = "MASK ATTACK"
    else:
        symbol = "⚠️"
        status = classification.upper().replace('_', ' ')
    
    print(f"\n{symbol} Classification: {status}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"Spoof Score: {results.get('spoof_score', 0):.3f} (0=live, 1=spoofed)")
    
    if 'frames_analyzed' in results:
        print(f"\nFrames Analyzed: {results['frames_analyzed']}")
        print(f"Faces Detected: {results['faces_detected']}")
    
    print("\n" + "-"*60)
    print("REASONING:")
    print("-"*60)
    reasoning = results.get('reasoning', {})
    if reasoning:
        for key, value in reasoning.items():
            print(f"  • {value}")
    else:
        print("  No specific reasoning provided")
    
    print("\n" + "-"*60)
    print("KEY FEATURES:")
    print("-"*60)
    features = results.get('features', {})
    print(f"  Texture Variance: {features.get('texture_variance', 0):.6f}")
    print(f"  Texture Entropy: {features.get('texture_entropy', 0):.4f}")
    print(f"  Motion Consistency: {features.get('motion_consistency', 0):.4f}")
    print(f"  Optical Flow: {features.get('optical_flow_score', 0):.4f}")
    print(f"  Blink Count: {features.get('blink_count', 0)}")
    print(f"  Has Blinks: {features.get('has_blinks', False)}")
    print(f"  Moiré Score: {features.get('moire_score', 0):.4f}")
    print(f"  High Freq Ratio: {features.get('high_freq_ratio', 0):.4f}")
    print(f"  Color Diversity: {features.get('color_diversity', 0):.2f}")
    print(f"  LBP Uniformity: {features.get('lbp_uniformity', 0):.4f}")
    
    if results.get('warning'):
        print("\n" + "-"*60)
        print(f"⚠️  WARNING: {results['warning']}")
    
    print("\n" + "="*60)


# Example usage
if __name__ == "__main__":
    # Initialize the application
    app = AntiSpoofingDetector()
    
    # Example: Process a video file
    video_file = "sample_video.mp4"  # Replace with your video path
    
    print("Video Anti-Spoofing System")
    print("Detects: Photos, Video Replays, Masks, and Real Faces\n")
    
    # You can also accept user input
    import sys
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    else:
        video_file = input("Enter video file path (or press Enter for demo mode): ").strip()
    
    if video_file and os.path.exists(video_file):
        # Process the video
        results = app.analyze_video(video_file)
        
        # Display results
        print_results(results)
        
        # Save results to file
        output_file = f"antispoofing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(results, output_file)
    else:
        print("\n📋 DEMO MODE - How to use:")
        print("="*60)
        print("1. Save this script as 'anti_spoofing.py'")
        print("2. Run: python anti_spoofing.py <video_file_path>")
        print("   OR")
        print("   python anti_spoofing.py")
        print("   (then enter the video path when prompted)")
        print("\nThe system will analyze the video and classify it as:")
        print("  • REAL - Live person")
        print("  • PHOTO - Printed photo or static image")
        print("  • VIDEO_REPLAY - Video played on screen")
        print("  • MASK - Person wearing a mask/3D model")
        print("="*60)