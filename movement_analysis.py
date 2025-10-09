"""
Advanced Movement Analysis Module for Stampede Detection
Implements involuntary flow, bottleneck movement, sudden acceleration, and wave-like motion detection
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import time
import math

class MovementAnalyzer:
    """Advanced movement analysis for crowd behavior detection"""
    
    def __init__(self, history_size: int = 30, flow_scale: float = 0.5):
        self.history_size = history_size
        self.flow_scale = flow_scale
        
        # Movement history buffers
        self.position_history = deque(maxlen=history_size)
        self.velocity_history = deque(maxlen=history_size)
        self.flow_history = deque(maxlen=history_size)
        self.density_history = deque(maxlen=history_size)
        
        # Analysis parameters
        self.domino_threshold = 0.7  # Threshold for domino-like motion
        self.bottleneck_threshold = 0.6  # Threshold for bottleneck detection
        self.panic_threshold = 2.0  # Threshold for sudden acceleration
        self.wave_threshold = 0.5  # Threshold for wave-like motion
        
        # Previous frame data for optical flow
        self.prev_gray = None
        self.prev_centers = None
        
    def compute_optical_flow(self, frame: np.ndarray, centers: List[Tuple[int, int]]) -> np.ndarray:
        """Compute optical flow for movement analysis with error handling"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.prev_gray is None:
                self.prev_gray = gray
                return np.zeros((len(centers), 1, 2), dtype=np.float32)
            
            if len(centers) == 0:
                self.prev_gray = gray
                return np.zeros((0, 1, 2), dtype=np.float32)
            
            # Ensure centers are valid and within frame bounds
            h, w = gray.shape
            valid_centers = []
            for cx, cy in centers:
                if 0 <= cx < w and 0 <= cy < h:
                    valid_centers.append([cx, cy])
            
            if len(valid_centers) == 0:
                self.prev_gray = gray
                return np.zeros((0, 1, 2), dtype=np.float32)
            
            # Convert to proper format for Lucas-Kanade
            points = np.array(valid_centers, dtype=np.float32).reshape(-1, 1, 2)
            
            # Ensure both images have the same size
            if self.prev_gray.shape != gray.shape:
                self.prev_gray = cv2.resize(self.prev_gray, (gray.shape[1], gray.shape[0]))
            
            # Compute optical flow using Lucas-Kanade method with error handling
            try:
                flow, status, error = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, 
                    points,
                    None,
                    winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                
                # Filter out failed tracks
                if flow is not None and status is not None:
                    valid_flow = flow[status.ravel() == 1]
                    if len(valid_flow) > 0:
                        self.prev_gray = gray
                        return valid_flow
                
            except cv2.error as e:
                print(f"[Movement Analysis] OpenCV error in optical flow: {e}")
                # Fallback: return zero flow
                self.prev_gray = gray
                return np.zeros((len(valid_centers), 1, 2), dtype=np.float32)
            
            self.prev_gray = gray
            return np.zeros((len(valid_centers), 1, 2), dtype=np.float32)
            
        except Exception as e:
            print(f"[Movement Analysis] Error in optical flow computation: {e}")
            self.prev_gray = gray if 'gray' in locals() else self.prev_gray
            return np.zeros((len(centers), 1, 2), dtype=np.float32)
    
    def analyze_involuntary_flow(self, centers: List[Tuple[int, int]], 
                               flow: np.ndarray, density_map: np.ndarray) -> Dict:
        """
        Detect involuntary flow (domino-like motion) where people are pushed
        without their own volition, creating cascading movement patterns
        """
        if len(centers) < 3 or flow is None:
            return {'involuntary_flow': False, 'flow_intensity': 0.0, 'cascade_direction': None}
        
        # Calculate movement vectors for each person
        movement_vectors = []
        if flow is not None and len(flow) > 0:
            # Ensure we don't exceed array bounds
            min_length = min(len(centers), len(flow))
            for i in range(min_length):
                center = centers[i]
                prev_pos = np.array(center)
                if len(flow[i].shape) > 1:
                    curr_pos = flow[i].flatten()
                else:
                    curr_pos = flow[i]
                if len(curr_pos) == 2:
                    movement = curr_pos - prev_pos
                    movement_vectors.append(movement)
        
        if len(movement_vectors) < 3:
            return {'involuntary_flow': False, 'flow_intensity': 0.0, 'cascade_direction': None}
        
        movement_vectors = np.array(movement_vectors)
        
        # Analyze cascade patterns - involuntary flow shows strong directional consistency
        # and spatial correlation (people near each other move in similar directions)
        cascade_score = 0.0
        direction_consistency = 0.0
        spatial_correlation = 0.0
        
        if len(movement_vectors) > 1:
            # Calculate direction consistency
            magnitudes = np.linalg.norm(movement_vectors, axis=1)
            if np.sum(magnitudes) > 0:
                normalized_vectors = movement_vectors / (magnitudes[:, np.newaxis] + 1e-6)
                # Check if most people are moving in similar directions
                mean_direction = np.mean(normalized_vectors, axis=0)
                direction_consistency = np.mean([np.dot(v, mean_direction) for v in normalized_vectors])
            
            # Calculate spatial correlation - people close together should move similarly
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    distance = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                    if distance < 100:  # Within reasonable distance
                        movement_similarity = np.dot(movement_vectors[i], movement_vectors[j]) / (
                            np.linalg.norm(movement_vectors[i]) * np.linalg.norm(movement_vectors[j]) + 1e-6
                        )
                        spatial_correlation += movement_similarity
        
        # Normalize spatial correlation
        if len(centers) > 1:
            spatial_correlation /= (len(centers) * (len(centers) - 1) / 2)
        
        # Calculate overall cascade score
        cascade_score = (direction_consistency * 0.6 + spatial_correlation * 0.4)
        
        # Determine cascade direction
        cascade_direction = None
        if cascade_score > self.domino_threshold:
            mean_movement = np.mean(movement_vectors, axis=0)
            if np.linalg.norm(mean_movement) > 5:  # Significant movement
                angle = math.atan2(mean_movement[1], mean_movement[0])
                if -math.pi/4 <= angle <= math.pi/4:
                    cascade_direction = 'right'
                elif math.pi/4 < angle <= 3*math.pi/4:
                    cascade_direction = 'down'
                elif -3*math.pi/4 <= angle < -math.pi/4:
                    cascade_direction = 'up'
                else:
                    cascade_direction = 'left'
        
        return {
            'involuntary_flow': bool(cascade_score > self.domino_threshold),
            'flow_intensity': float(cascade_score),
            'cascade_direction': str(cascade_direction) if cascade_direction else None,
            'direction_consistency': float(direction_consistency),
            'spatial_correlation': float(spatial_correlation)
        }
    
    def analyze_bottleneck_movement(self, centers: List[Tuple[int, int]], 
                                  density_map: np.ndarray, frame_shape: Tuple[int, int]) -> Dict:
        """
        Detect bottleneck movement where crowd density creates restricted flow
        and directional movement patterns
        """
        if len(centers) < 5 or density_map.size == 0:
            return {'bottleneck': False, 'bottleneck_intensity': 0.0, 'flow_direction': None}
        
        h, w = frame_shape[:2]
        grid_h, grid_w = density_map.shape
        
        # Find high-density regions (potential bottlenecks)
        high_density_threshold = np.percentile(density_map, 80)
        high_density_mask = density_map > high_density_threshold
        
        # Analyze flow direction in high-density areas
        flow_directions = []
        density_gradients = []
        
        # Calculate density gradients to find flow direction
        grad_x = cv2.Sobel(density_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(density_map, cv2.CV_64F, 0, 1, ksize=3)
        
        # Find dominant flow direction in high-density areas
        high_density_grad_x = grad_x[high_density_mask]
        high_density_grad_y = grad_y[high_density_mask]
        
        if len(high_density_grad_x) > 0:
            mean_grad_x = np.mean(high_density_grad_x)
            mean_grad_y = np.mean(high_density_grad_y)
            
            # Calculate flow direction from density gradient
            if abs(mean_grad_x) > 0.1 or abs(mean_grad_y) > 0.1:
                flow_angle = math.atan2(mean_grad_y, mean_grad_x)
                flow_directions.append(flow_angle)
        
        # Analyze people movement in relation to density gradients
        bottleneck_score = 0.0
        flow_direction = None
        
        if flow_directions:
            # Calculate consistency of flow direction
            flow_consistency = 1.0 - np.std(flow_directions) / math.pi
            
            # Calculate bottleneck intensity based on density concentration
            density_concentration = np.sum(high_density_mask) / (grid_h * grid_w)
            
            # Calculate flow restriction (how much movement is constrained)
            flow_restriction = min(1.0, density_concentration * 2)
            
            bottleneck_score = (flow_consistency * 0.4 + 
                              density_concentration * 0.3 + 
                              flow_restriction * 0.3)
            
            # Determine flow direction
            if bottleneck_score > self.bottleneck_threshold:
                mean_flow_angle = np.mean(flow_directions)
                if -math.pi/4 <= mean_flow_angle <= math.pi/4:
                    flow_direction = 'right'
                elif math.pi/4 < mean_flow_angle <= 3*math.pi/4:
                    flow_direction = 'down'
                elif -3*math.pi/4 <= mean_flow_angle < -math.pi/4:
                    flow_direction = 'up'
                else:
                    flow_direction = 'left'
        
        return {
            'bottleneck': bool(bottleneck_score > self.bottleneck_threshold),
            'bottleneck_intensity': float(bottleneck_score),
            'flow_direction': str(flow_direction) if flow_direction else None,
            'density_concentration': float(np.sum(high_density_mask) / (grid_h * grid_w)) if grid_h * grid_w > 0 else 0.0
        }
    
    def analyze_sudden_acceleration(self, centers: List[Tuple[int, int]], 
                                  flow: np.ndarray) -> Dict:
        """
        Detect sudden acceleration or panic triggers where people suddenly
        increase their movement speed or change direction rapidly
        """
        if len(centers) < 3 or flow is None:
            return {'sudden_acceleration': False, 'acceleration_intensity': 0.0, 'panic_level': 'low'}
        
        # Calculate current velocities
        current_velocities = []
        if flow is not None and len(flow) > 0:
            # Ensure we don't exceed array bounds
            min_length = min(len(centers), len(flow))
            for i in range(min_length):
                center = centers[i]
                prev_pos = np.array(center)
                if len(flow[i].shape) > 1:
                    curr_pos = flow[i].flatten()
                else:
                    curr_pos = flow[i]
                if len(curr_pos) == 2:
                    velocity = curr_pos - prev_pos
                    current_velocities.append(np.linalg.norm(velocity))
        
        if len(current_velocities) < 3:
            return {'sudden_acceleration': False, 'acceleration_intensity': 0.0, 'panic_level': 'low'}
        
        # Store current velocities
        self.velocity_history.append(current_velocities)
        
        if len(self.velocity_history) < 3:
            return {'sudden_acceleration': False, 'acceleration_intensity': 0.0, 'panic_level': 'low'}
        
        # Calculate acceleration (change in velocity)
        prev_velocities = self.velocity_history[-2]
        curr_velocities = self.velocity_history[-1]
        
        accelerations = []
        for i in range(min(len(prev_velocities), len(curr_velocities))):
            acceleration = curr_velocities[i] - prev_velocities[i]
            accelerations.append(acceleration)
        
        if not accelerations:
            return {'sudden_acceleration': False, 'acceleration_intensity': 0.0, 'panic_level': 'low'}
        
        # Calculate acceleration statistics
        mean_acceleration = np.mean(accelerations)
        max_acceleration = np.max(accelerations)
        acceleration_std = np.std(accelerations)
        
        # Detect sudden acceleration
        sudden_acceleration = max_acceleration > self.panic_threshold
        
        # Calculate panic level based on acceleration patterns
        panic_level = 'low'
        if max_acceleration > self.panic_threshold * 2:
            panic_level = 'high'
        elif max_acceleration > self.panic_threshold:
            panic_level = 'moderate'
        
        # Calculate acceleration intensity
        acceleration_intensity = min(1.0, max_acceleration / (self.panic_threshold * 2))
        
        return {
            'sudden_acceleration': bool(sudden_acceleration),
            'acceleration_intensity': float(acceleration_intensity),
            'panic_level': str(panic_level),
            'mean_acceleration': float(mean_acceleration),
            'max_acceleration': float(max_acceleration)
        }
    
    def analyze_wave_motion(self, centers: List[Tuple[int, int]], 
                          density_map: np.ndarray, frame_shape: Tuple[int, int]) -> Dict:
        """
        Detect wave-like motion patterns in crowd movement
        """
        if len(centers) < 5 or density_map.size == 0:
            return {'wave_motion': False, 'wave_intensity': 0.0, 'wave_direction': None}
        
        # Store current density map
        self.density_history.append(density_map.copy())
        
        if len(self.density_history) < 5:
            return {'wave_motion': False, 'wave_intensity': 0.0, 'wave_direction': None}
        
        # Analyze density wave propagation
        wave_scores = []
        wave_directions = []
        
        # Look for wave patterns in density changes
        for i in range(1, len(self.density_history)):
            prev_density = self.density_history[i-1]
            curr_density = self.density_history[i]
            
            # Calculate density change
            density_change = curr_density - prev_density
            
            # Find wave-like patterns using spatial correlation
            # A wave should show smooth transitions across space
            grad_x = cv2.Sobel(density_change, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(density_change, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate wave strength (smoothness of density change)
            wave_strength = np.mean(np.abs(grad_x) + np.abs(grad_y))
            
            # Calculate dominant wave direction
            if np.sum(np.abs(grad_x)) > 0 or np.sum(np.abs(grad_y)) > 0:
                mean_grad_x = np.mean(grad_x)
                mean_grad_y = np.mean(grad_y)
                wave_angle = math.atan2(mean_grad_y, mean_grad_x)
                wave_directions.append(wave_angle)
            
            wave_scores.append(wave_strength)
        
        if not wave_scores:
            return {'wave_motion': False, 'wave_intensity': 0.0, 'wave_direction': None}
        
        # Calculate overall wave intensity
        wave_intensity = np.mean(wave_scores)
        
        # Determine wave direction
        wave_direction = None
        if wave_directions and wave_intensity > self.wave_threshold:
            mean_wave_angle = np.mean(wave_directions)
            if -math.pi/4 <= mean_wave_angle <= math.pi/4:
                wave_direction = 'right'
            elif math.pi/4 < mean_wave_angle <= 3*math.pi/4:
                wave_direction = 'down'
            elif -3*math.pi/4 <= mean_wave_angle < -math.pi/4:
                wave_direction = 'up'
            else:
                wave_direction = 'left'
        
        return {
            'wave_motion': bool(wave_intensity > self.wave_threshold),
            'wave_intensity': float(wave_intensity),
            'wave_direction': str(wave_direction) if wave_direction else None,
            'wave_consistency': float(1.0 - np.std(wave_directions) / math.pi) if wave_directions else 0.0
        }
    
    def analyze_movement_patterns(self, frame: np.ndarray, centers: List[Tuple[int, int]], 
                                density_map: np.ndarray) -> Dict:
        """
        Comprehensive movement analysis combining all detection methods
        """
        # Compute optical flow
        flow = self.compute_optical_flow(frame, centers)
        
        # Analyze all movement patterns
        involuntary_flow = self.analyze_involuntary_flow(centers, flow, density_map)
        bottleneck_movement = self.analyze_bottleneck_movement(centers, density_map, frame.shape)
        sudden_acceleration = self.analyze_sudden_acceleration(centers, flow)
        wave_motion = self.analyze_wave_motion(centers, density_map, frame.shape)
        
        # Calculate overall movement risk score
        risk_factors = []
        risk_score = 0.0
        
        if involuntary_flow['involuntary_flow']:
            risk_factors.append('involuntary_flow')
            risk_score += 0.3
        
        if bottleneck_movement['bottleneck']:
            risk_factors.append('bottleneck_movement')
            risk_score += 0.25
        
        if sudden_acceleration['sudden_acceleration']:
            risk_factors.append('sudden_acceleration')
            risk_score += 0.3
        
        if wave_motion['wave_motion']:
            risk_factors.append('wave_motion')
            risk_score += 0.15
        
        # Determine overall movement risk level
        if risk_score >= 0.7:
            movement_risk_level = 'critical'
        elif risk_score >= 0.5:
            movement_risk_level = 'high'
        elif risk_score >= 0.3:
            movement_risk_level = 'moderate'
        else:
            movement_risk_level = 'low'
        
        return {
            'involuntary_flow': involuntary_flow,
            'bottleneck_movement': bottleneck_movement,
            'sudden_acceleration': sudden_acceleration,
            'wave_motion': wave_motion,
            'movement_risk_score': float(risk_score),
            'movement_risk_level': str(movement_risk_level),
            'movement_risk_factors': [str(factor) for factor in risk_factors],
            'timestamp': float(time.time())
        }
    
    def reset_history(self):
        """Reset all history buffers"""
        self.position_history.clear()
        self.velocity_history.clear()
        self.flow_history.clear()
        self.density_history.clear()
        self.prev_gray = None
        self.prev_centers = None
