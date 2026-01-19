"""
Anti-spoofing feedback system
Records feedback and automatically adjusts configuration based on errors
"""

import json
import os
from datetime import datetime
from typing import Dict, List
import shutil


class AntispoofingFeedbackSystem:
    def __init__(self, config_path: str, feedback_log_path: str = None):
        self.config_path = config_path
        
        if feedback_log_path is None:
            feedback_log_path = os.path.join(
                os.path.dirname(config_path), 
                'antispoofing_feedback.json'
            )
        self.feedback_log_path = feedback_log_path
        
        # Load existing feedback log or create new
        self.feedback_log = self._load_feedback_log()
    
    def _load_feedback_log(self) -> List[Dict]:
        """Load existing feedback log"""
        if os.path.exists(self.feedback_log_path):
            with open(self.feedback_log_path, 'r') as f:
                return json.load(f)
        return []
    
    def _save_feedback_log(self):
        """Save feedback log to file"""
        with open(self.feedback_log_path, 'w') as f:
            json.dump(self.feedback_log, f, indent=2)
    
    def record_feedback(self, 
                       detection_result: Dict, 
                       is_correct: bool, 
                       true_label: str = None) -> Dict:
        """
        Record user feedback about a detection
        
        Args:
            detection_result: The result from antispoofing detector
            is_correct: Whether the detection was correct
            true_label: What the true label should have been (if incorrect)
        
        Returns:
            Feedback record with suggested adjustments
        """
        feedback_record = {
            'timestamp': datetime.now().isoformat(),
            'detection_result': detection_result,
            'is_correct': is_correct,
            'true_label': true_label
        }
        
        # If incorrect, analyze what went wrong
        if not is_correct:
            adjustments = self._analyze_error(detection_result, true_label)
            feedback_record['suggested_adjustments'] = adjustments
        
        self.feedback_log.append(feedback_record)
        self._save_feedback_log()
        
        return feedback_record
    
    def _analyze_error(self, detection_result: Dict, true_label: str) -> Dict:
        """
        Analyze what caused the misclassification
        
        Returns dict with features to adjust and by how much
        """
        adjustments = {
            'weights_to_decrease': [],
            'weights_to_increase': [],
            'thresholds_to_adjust': []
        }
        
        predicted = detection_result.get('classification', 'unknown')
        reasoning = detection_result.get('reasoning', {})
        features = detection_result.get('features', {})
        
        # Identify which features contributed most to wrong decision
        # These are the features that triggered reasoning for the wrong class
        
        # If we predicted 'real' but it was actually a spoof
        if predicted == 'real' and true_label != 'real':
            # Decrease weights that made us think it was real
            for key in reasoning.keys():
                if any(term in key for term in ['blink', 'motion', 'texture', 'color', 'optical']):
                    adjustments['weights_to_decrease'].append({
                        'category': 'real',
                        'feature': key,
                        'reason': f"Contributed to false 'real' classification"
                    })
            
            # Increase weights for actual spoof type detection
            if true_label in ['photo', 'video_replay', 'mask']:
                adjustments['weights_to_increase'].append({
                    'category': true_label,
                    'reason': f"Should have detected {true_label}"
                })
        
        # If we predicted spoof but it was actually real
        elif predicted != 'real' and true_label == 'real':
            # Decrease weights that made us think it was spoofed
            for key in reasoning.keys():
                adjustments['weights_to_decrease'].append({
                    'category': predicted,
                    'feature': key,
                    'reason': f"Contributed to false '{predicted}' classification"
                })
            
            # Increase real weights
            adjustments['weights_to_increase'].append({
                'category': 'real',
                'reason': "Should have detected as real"
            })
        
        # If we predicted wrong type of spoof
        elif predicted != 'real' and true_label != 'real' and predicted != true_label:
            # Decrease wrong spoof type weights
            for key in reasoning.keys():
                adjustments['weights_to_decrease'].append({
                    'category': predicted,
                    'feature': key,
                    'reason': f"Contributed to false '{predicted}' classification"
                })
            
            # Increase correct spoof type weights
            adjustments['weights_to_increase'].append({
                'category': true_label,
                'reason': f"Should have detected as {true_label}"
            })
        
        return adjustments
    
    def apply_automatic_adjustment(self, feedback_record: Dict, adjustment_factor: float = 0.10):
        """
        Automatically adjust config based on feedback
        
        Args:
            feedback_record: The feedback record with suggested adjustments
            adjustment_factor: Percentage to adjust weights (default 10%)
        """
        if feedback_record.get('is_correct', True):
            return {"message": "No adjustment needed for correct detection"}
        
        adjustments = feedback_record.get('suggested_adjustments', {})
        if not adjustments:
            return {"message": "No adjustments suggested"}
        
        # Backup current config
        self._backup_config()
        
        # Load current config
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        changes_made = []
        
        # Apply weight decreases
        for adj in adjustments.get('weights_to_decrease', []):
            category = adj['category']
            feature = adj.get('feature')
            
            if category in config['weights']:
                # If specific feature mentioned, adjust that
                if feature and feature in config['weights'][category]:
                    old_value = config['weights'][category][feature]
                    new_value = old_value * (1 - adjustment_factor)
                    config['weights'][category][feature] = round(new_value, 2)
                    changes_made.append({
                        'type': 'weight_decrease',
                        'path': f"weights.{category}.{feature}",
                        'old_value': old_value,
                        'new_value': new_value,
                        'reason': adj['reason']
                    })
                else:
                    # Decrease all weights in this category proportionally
                    for key in config['weights'][category]:
                        old_value = config['weights'][category][key]
                        new_value = old_value * (1 - adjustment_factor)
                        config['weights'][category][key] = round(new_value, 2)
                        changes_made.append({
                            'type': 'weight_decrease',
                            'path': f"weights.{category}.{key}",
                            'old_value': old_value,
                            'new_value': new_value,
                            'reason': adj['reason']
                        })
        
        # Apply weight increases
        for adj in adjustments.get('weights_to_increase', []):
            category = adj['category']
            
            if category in config['weights']:
                # Increase all weights in this category
                for key in config['weights'][category]:
                    old_value = config['weights'][category][key]
                    new_value = old_value * (1 + adjustment_factor)
                    config['weights'][category][key] = round(new_value, 2)
                    changes_made.append({
                        'type': 'weight_increase',
                        'path': f"weights.{category}.{key}",
                        'old_value': old_value,
                        'new_value': new_value,
                        'reason': adj['reason']
                    })
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return {
            'success': True,
            'changes_made': changes_made,
            'backup_path': self.config_path + '.backup',
            'adjustment_factor': adjustment_factor
        }
    
    def _backup_config(self):
        """Create backup of current config"""
        backup_path = self.config_path + '.backup'
        shutil.copy2(self.config_path, backup_path)
    
    def get_feedback_stats(self) -> Dict:
        """Get statistics about feedback"""
        total = len(self.feedback_log)
        correct = sum(1 for f in self.feedback_log if f.get('is_correct', False))
        incorrect = total - correct
        
        return {
            'total_feedback': total,
            'correct_detections': correct,
            'incorrect_detections': incorrect,
            'accuracy': correct / total if total > 0 else 0
        }
