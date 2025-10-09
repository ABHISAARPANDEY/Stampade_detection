import os
import json
import time
import threading
import base64
from typing import Dict, List, Optional
from collections import deque
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import tempfile
import uuid
import torch
from movement_analysis import MovementAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'stampede_detection_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for video processing and advanced features
current_model = None
processing_thread = None
is_processing = False
latest_frame = None
frame_queue = deque(maxlen=1)

# Advanced features global variables
crowd_flow_history = deque(maxlen=30)
risk_factors = {
    'density_trend': deque(maxlen=10),
    'people_trend': deque(maxlen=10),
    'movement_intensity': deque(maxlen=10)
}
last_alert_time = 0

# Movement analyzer instance
movement_analyzer = None

detection_results = {
    'people_count': 0,
    'density': 0.0,
    'status': 'SAFE',
    'status_color': (0, 200, 0),
    'alerts': [],
    'flow_data': {'flow_intensity': 0.0, 'movement_direction': 'stable'},
    'risk_assessment': {'risk_score': 0.0, 'risk_level': 'low'},
    'movement_analysis': {'movement_risk_level': 'low', 'movement_risk_score': 0.0, 'movement_risk_factors': []}
}

# Enhanced Configuration for better dense crowd detection
DEFAULT_AREA_M2 = 25.0
DEFAULT_CONFIDENCE = 0.15  # Lower confidence for better detection in dense crowds
DEFAULT_GRID_W = 32  # Increased for finer analysis
DEFAULT_GRID_H = 24  # Increased for finer analysis
DANGER_DENSITY = 6.0  # User requirement: 6 people/m² for stampede
WARNING_DENSITY = 4.0  # User requirement: 4 people/m² for crowded
DEFAULT_IMAGE_SIZE = 1280  # Increased for better accuracy

# GPU Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0

def select_best_model():
    """Select the best available YOLO model for accuracy"""
    candidates = [
        "./training/yolov8l/train/weights/best.pt",
        "./training/yolov8m/train/weights/best.pt", 
        "./training/yolov8s/train/weights/best.pt",
        "./training/yolov8n/train/weights/best.pt",
        "./yolov8l.pt",
        "./yolov8m.pt",
        "./yolov8s.pt",
        "./yolov8n.pt",
    ]
    
    for model_path in candidates:
        if os.path.exists(model_path):
            return model_path
    
    return "yolov8l.pt"  # Default to large model

def initialize_gpu_model(model_path):
    """Initialize YOLO model with GPU acceleration"""
    global DEVICE, GPU_COUNT
    
    print(f"🚀 Initializing YOLO model with GPU acceleration...")
    print(f"📱 Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"🔢 GPU Count: {GPU_COUNT}")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Load model
    model = YOLO(model_path)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model.to(DEVICE)
        print(f"✅ Model loaded on GPU: {DEVICE}")
    else:
        print("✅ Model loaded on CPU")
    
    return model

def compute_density_map(centers, frame_shape, grid_w, grid_h, total_area_m2):
    """Enhanced crowd density computation with better accuracy for dense crowds."""
    h, w = frame_shape[:2]
    density_count = np.zeros((grid_h, grid_w), dtype=np.float32)
    if not centers:
        return density_count
    
    cell_w = max(1, w // grid_w)
    cell_h = max(1, h // grid_h)
    
    # Enhanced counting with weighted distribution for overlapping detections
    for cx, cy in centers:
        # Find primary grid cell
        gx = min(grid_w - 1, max(0, cx // cell_w))
        gy = min(grid_h - 1, max(0, cy // cell_h))
        density_count[gy, gx] += 1.0
        
        # Add weighted contribution to neighboring cells for better density estimation
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ngx = gx + dx
                ngy = gy + dy
                if 0 <= ngx < grid_w and 0 <= ngy < grid_h:
                    # Calculate distance from person center to neighboring cell
                    cell_center_x = ngx * cell_w + cell_w // 2
                    cell_center_y = ngy * cell_h + cell_h // 2
                    distance = np.sqrt((cx - cell_center_x)**2 + (cy - cell_center_y)**2)
                    max_distance = np.sqrt(cell_w**2 + cell_h**2)
                    
                    # Weight decreases with distance
                    weight = max(0, 1.0 - distance / max_distance) * 0.1  # Small contribution
                    density_count[ngy, ngx] += weight
    
    # Convert to people per square meter
    total_cells = grid_w * grid_h
    area_per_cell_m2 = total_area_m2 / total_cells
    area_per_cell_m2 = max(area_per_cell_m2, 0.05)  # Reduced minimum for better sensitivity
    
    density_per_m2 = density_count / area_per_cell_m2
    density_per_m2 = cv2.GaussianBlur(density_per_m2, (5, 5), 1.0)  # Enhanced smoothing
    return density_per_m2

def analyze_crowd_flow(centers, frame_shape):
    """Analyze crowd movement patterns and flow intensity."""
    global crowd_flow_history
    
    if len(centers) < 2:
        return {'flow_intensity': 0.0, 'movement_direction': 'stable', 'crowd_velocity': 0.0}
    
    # Store current centers
    crowd_flow_history.append({
        'centers': centers.copy(),
        'timestamp': time.time(),
        'frame_shape': frame_shape
    })
    
    if len(crowd_flow_history) < 2:
        return {'flow_intensity': 0.0, 'movement_direction': 'stable', 'crowd_velocity': 0.0}
    
    # Calculate movement between frames
    prev_data = crowd_flow_history[-2]
    curr_data = crowd_flow_history[-1]
    
    prev_centers = prev_data['centers']
    curr_centers = curr_data['centers']
    
    # Simple movement analysis
    total_movement = 0.0
    movement_count = 0
    
    for curr_center in curr_centers:
        min_distance = float('inf')
        for prev_center in prev_centers:
            distance = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
            min_distance = min(min_distance, distance)
        
        if min_distance < 100:  # Reasonable movement threshold
            total_movement += min_distance
            movement_count += 1
    
    # Calculate flow intensity
    avg_movement = total_movement / max(movement_count, 1)
    flow_intensity = min(avg_movement / 50.0, 1.0)  # Normalize to 0-1
    
    # Determine movement direction
    if flow_intensity > 0.7:
        movement_direction = 'high_movement'
    elif flow_intensity > 0.3:
        movement_direction = 'moderate_movement'
    else:
        movement_direction = 'stable'
    
    return {
        'flow_intensity': flow_intensity,
        'movement_direction': movement_direction,
        'crowd_velocity': avg_movement,
        'movement_count': movement_count
    }

def assess_risk_factors(num_people, overall_density, max_density, flow_data):
    """Advanced risk assessment using multiple factors."""
    global risk_factors
    
    # Update trend data
    risk_factors['density_trend'].append(overall_density)
    risk_factors['people_trend'].append(num_people)
    risk_factors['movement_intensity'].append(flow_data['flow_intensity'])
    
    # Calculate trends
    density_trend = 0.0
    people_trend = 0.0
    
    if len(risk_factors['density_trend']) >= 3:
        recent_density = list(risk_factors['density_trend'])[-3:]
        density_trend = (recent_density[-1] - recent_density[0]) / max(recent_density[0], 0.1)
    
    if len(risk_factors['people_trend']) >= 3:
        recent_people = list(risk_factors['people_trend'])[-3:]
        people_trend = (recent_people[-1] - recent_people[0]) / max(recent_people[0], 1)
    
    # More realistic risk scoring
    risk_score = 0.0
    risk_factors_list = []
    
    # Density factor (more conservative)
    if overall_density >= DANGER_DENSITY and num_people >= 10:
        risk_score += 0.4
        risk_factors_list.append('high_density')
    elif overall_density >= WARNING_DENSITY and num_people >= 6:
        risk_score += 0.2
        risk_factors_list.append('moderate_density')
    
    # People count factor (more conservative)
    if num_people >= 15:
        risk_score += 0.3
        risk_factors_list.append('many_people')
    elif num_people >= 8:
        risk_score += 0.1
        risk_factors_list.append('moderate_people')
    
    # Movement factor (more conservative)
    if flow_data['flow_intensity'] > 0.8 and num_people >= 8:
        risk_score += 0.2
        risk_factors_list.append('high_movement')
    elif flow_data['flow_intensity'] > 0.5 and num_people >= 5:
        risk_score += 0.1
        risk_factors_list.append('moderate_movement')
    
    # Trend factors (more conservative)
    if density_trend > 0.8 and num_people >= 8:
        risk_score += 0.1
        risk_factors_list.append('increasing_density')
    
    if people_trend > 0.5 and num_people >= 8:
        risk_score += 0.1
        risk_factors_list.append('increasing_crowd')
    
    # Determine overall risk level (more conservative)
    if risk_score >= 0.8:
        risk_level = 'critical'
    elif risk_score >= 0.6:
        risk_level = 'high'
    elif risk_score >= 0.4:
        risk_level = 'moderate'
    else:
        risk_level = 'low'
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_factors': risk_factors_list,
        'density_trend': density_trend,
        'people_trend': people_trend
    }

def process_frame(frame, model, area_m2, confidence, grid_w, grid_h):
    """Enhanced frame processing with better dense crowd detection and advanced features"""
    global detection_results
    
    # Resize frame for processing to improve performance while maintaining accuracy
    original_shape = frame.shape
    if frame.shape[1] > 1280:  # Resize if too large
        scale = 1280 / frame.shape[1]
        new_width = 1280
        new_height = int(frame.shape[0] * scale)
        frame_resized = cv2.resize(frame, (new_width, new_height))
    else:
        frame_resized = frame
        scale = 1.0
    
    # Enhanced YOLOv11 detection with GPU acceleration and optimized settings for best accuracy
    results = model(frame_resized, 
                   conf=confidence, 
                   classes=[0], 
                   verbose=False,
                   imgsz=min(1280, max(frame_resized.shape[0], frame_resized.shape[1])),  # Adaptive size
                   iou=0.25,  # Lower IoU for better dense crowd detection
                   max_det=3000,  # Higher limit for dense crowds with YOLOv11
                   agnostic_nms=True,  # Better NMS for dense crowds
                   augment=True,  # Enable augmentation for YOLOv11's superior accuracy
                   device=DEVICE,  # Use GPU/CPU device
                   half=True if DEVICE == 'cuda' else False,  # Use FP16 on GPU for speed
                   save=False,
                   save_txt=False,
                   save_conf=False)
    
    # Enhanced person detection extraction
    centers = []
    detection_boxes = []
    confidence_scores = []
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else None
        conf = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else None
        
        for i, box in enumerate(xyxy):
            if cls is not None and int(cls[i]) != 0:
                continue
            
            # Additional confidence filtering
            if conf is not None and conf[i] < confidence:
                continue
                
            x0, y0, x1, y1 = box.astype(int)
            
            # Scale coordinates back to original frame size if resized
            if scale != 1.0:
                x0 = int(x0 / scale)
                y0 = int(y0 / scale)
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
            
            # Ensure boxes are within original frame bounds
            h, w = original_shape[:2]
            x0 = max(0, min(w-1, x0))
            y0 = max(0, min(h-1, y0))
            x1 = max(0, min(w-1, x1))
            y1 = max(0, min(h-1, y1))
            
            # Skip invalid boxes
            if x1 <= x0 or y1 <= y0:
                continue
                
            cx = int((x0 + x1) * 0.5)
            cy = int((y0 + y1) * 0.5)
            
            centers.append((cx, cy))
            detection_boxes.append((x0, y0, x1, y1))
            confidence_scores.append(float(conf[i]) if conf is not None else 1.0)
    
    # Calculate density
    density_map = compute_density_map(centers, frame.shape, grid_w, grid_h, area_m2)
    overall_density = len(centers) / area_m2 if area_m2 > 0 else 0.0
    
    # Get number of people detected
    num_people = len(centers)
    max_density = float(np.max(density_map)) if density_map.size else 0.0
    avg_density = float(np.mean(density_map)) if density_map.size else 0.0
    
    # Advanced crowd flow analysis
    flow_data = analyze_crowd_flow(centers, frame.shape)
    
    # Advanced movement analysis (optimized - only run every few frames for performance)
    global movement_analyzer
    movement_analysis = {
        'movement_risk_level': 'low', 
        'movement_risk_score': 0.0, 
        'movement_risk_factors': [],
        'involuntary_flow': {'involuntary_flow': False, 'flow_intensity': 0.0, 'cascade_direction': None},
        'bottleneck_movement': {'bottleneck': False, 'bottleneck_intensity': 0.0, 'flow_direction': None},
        'sudden_acceleration': {'sudden_acceleration': False, 'acceleration_intensity': 0.0, 'panic_level': 'low'},
        'wave_motion': {'wave_motion': False, 'wave_intensity': 0.0, 'wave_direction': None}
    }
    
    # Only run movement analysis every 10th frame for better performance
    global frame_count_for_movement
    if 'frame_count_for_movement' not in globals():
        frame_count_for_movement = 0
    
    frame_count_for_movement += 1
    
    if len(centers) >= 3 and frame_count_for_movement % 10 == 0:  # Only analyze every 10th frame
        try:
            if movement_analyzer is None:
                movement_analyzer = MovementAnalyzer(history_size=10, flow_scale=0.5)  # Further reduced history size
            
            # Use original frame for movement analysis (not resized)
            movement_analysis = movement_analyzer.analyze_movement_patterns(frame, centers, density_map)
        except Exception as e:
            # Silently handle errors to avoid spam
            pass
    
    # Advanced risk assessment
    risk_assessment = assess_risk_factors(num_people, overall_density, max_density, flow_data)
    
    # Enhanced status determination with advanced risk assessment
    # (num_people, max_density, avg_density already calculated above)
    
    # Fixed risk assessment with realistic thresholds
    density_factor = max(overall_density, max_density)
    people_factor = num_people
    
    # User-specified thresholds: 6 people/m² for stampede, 4 for crowded
    # Use overall density for main status, max local density for alerts
    if overall_density >= DANGER_DENSITY:  # >= 6 people/m² overall
        status = "DANGER: STAMPEDE RISK"
        status_color = (0, 0, 255)  # Red
        alert_level = "danger"
    elif overall_density >= WARNING_DENSITY:  # >= 4 people/m² overall
        status = "CROWDED: MONITOR CLOSELY"
        status_color = (0, 255, 255)  # Yellow
        alert_level = "warning"
    else:
        status = "SAFE: NORMAL CONDITIONS"
        status_color = (0, 200, 0)  # Green
        alert_level = "safe"
    
    # Update global results with enhanced metrics and advanced features
    # Convert numpy types to Python native types for JSON serialization
    detection_results = {
        'people_count': int(num_people),
        'density': float(round(overall_density, 2)),
        'status': str(status),
        'status_color': list(status_color),
        'alert_level': str(alert_level),
        'max_density': float(round(max_density, 2)),
        'avg_density': float(round(avg_density, 2)),
        'confidence_scores': [float(score) for score in confidence_scores],
        'detection_boxes': [[int(x) for x in box] for box in detection_boxes],
        'flow_data': {
            'flow_intensity': float(flow_data.get('flow_intensity', 0.0)),
            'movement_direction': str(flow_data.get('movement_direction', 'stable')),
            'crowd_velocity': float(flow_data.get('crowd_velocity', 0.0)),
            'movement_count': int(flow_data.get('movement_count', 0))
        },
        'risk_assessment': {
            'risk_score': float(risk_assessment['risk_score']),
            'risk_level': str(risk_assessment['risk_level']),
            'risk_factors': [str(factor) for factor in risk_assessment['risk_factors']],
            'density_trend': float(risk_assessment['density_trend']),
            'people_trend': float(risk_assessment['people_trend'])
        },
        'movement_analysis': {
            'movement_risk_score': float(movement_analysis['movement_risk_score']),
            'movement_risk_level': str(movement_analysis['movement_risk_level']),
            'movement_risk_factors': [str(factor) for factor in movement_analysis['movement_risk_factors']],
            'involuntary_flow': {
                'involuntary_flow': bool(movement_analysis['involuntary_flow']['involuntary_flow']),
                'flow_intensity': float(movement_analysis['involuntary_flow']['flow_intensity']),
                'cascade_direction': str(movement_analysis['involuntary_flow']['cascade_direction']) if movement_analysis['involuntary_flow']['cascade_direction'] else None
            },
            'bottleneck_movement': {
                'bottleneck': bool(movement_analysis['bottleneck_movement']['bottleneck']),
                'bottleneck_intensity': float(movement_analysis['bottleneck_movement']['bottleneck_intensity']),
                'flow_direction': str(movement_analysis['bottleneck_movement']['flow_direction']) if movement_analysis['bottleneck_movement']['flow_direction'] else None
            },
            'sudden_acceleration': {
                'sudden_acceleration': bool(movement_analysis['sudden_acceleration']['sudden_acceleration']),
                'acceleration_intensity': float(movement_analysis['sudden_acceleration']['acceleration_intensity']),
                'panic_level': str(movement_analysis['sudden_acceleration']['panic_level'])
            },
            'wave_motion': {
                'wave_motion': bool(movement_analysis['wave_motion']['wave_motion']),
                'wave_intensity': float(movement_analysis['wave_motion']['wave_intensity']),
                'wave_direction': str(movement_analysis['wave_motion']['wave_direction']) if movement_analysis['wave_motion']['wave_direction'] else None
            }
        },
        'timestamp': float(time.time())
    }
    
    # Create visualization
    vis_frame = frame.copy()
    
    # Simple dot visualization for person detection
    if len(detection_boxes) > 0:
        cell_w = max(1, frame.shape[1] // grid_w)
        cell_h = max(1, frame.shape[0] // grid_h)
        
        for i, (x0, y0, x1, y1) in enumerate(detection_boxes):
            cx = int((x0 + x1) * 0.5)
            cy = int((y0 + y1) * 0.5)
            gx = min(grid_w - 1, max(0, cx // cell_w))
            gy = min(grid_h - 1, max(0, cy // cell_h))
            local_density = float(density_map[gy, gx]) if density_map.size else 0.0
            
            # Enhanced color coding based on both overall and local density
            if overall_density >= DANGER_DENSITY or local_density >= DANGER_DENSITY:
                color = (0, 0, 255)  # Red
                dot_size = 4  # Much smaller dots
            elif overall_density >= WARNING_DENSITY or local_density >= WARNING_DENSITY:
                color = (0, 255, 255)  # Yellow
                dot_size = 3  # Much smaller dots
            else:
                color = (0, 200, 0)  # Green
                dot_size = 2  # Much smaller dots
            
            # Draw simple clear dot at person center
            cv2.circle(vis_frame, (cx, cy), dot_size, color, -1)  # Filled circle
            cv2.circle(vis_frame, (cx, cy), dot_size, (255, 255, 255), 1)  # Thin white outline
            
            # Optional: Add small confidence indicator (very subtle)
            if confidence_scores[i] < 0.3:  # Only show for low confidence
                cv2.putText(vis_frame, f"{confidence_scores[i]:.1f}", (cx+15, cy-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Enhanced status overlay with upper bar and range information
    cv2.rectangle(vis_frame, (10, 10), (650, 160), (0, 0, 0), -1)
    cv2.rectangle(vis_frame, (10, 10), (650, 160), (255, 255, 255), 1)
    
    # Calculate range information
    range_low = max(0, num_people)
    range_high_10 = num_people + 10
    range_high_20 = num_people + 20
    
    # Calculate average confidence
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
    
    # Range color based on confidence
    if avg_confidence > 0.8:
        range_color = (0, 255, 0)  # Green - high confidence
        range_text = f"Range: {range_low}-{range_high_10} (±10)"
    elif avg_confidence > 0.6:
        range_color = (0, 255, 255)  # Yellow - medium confidence
        range_text = f"Range: {range_low}-{range_high_20} (±20)"
    else:
        range_color = (0, 165, 255)  # Orange - low confidence
        range_text = f"Range: {range_low}-{range_high_20} (±20) - Low Confidence"
    
    # Main metrics with range
    cv2.putText(vis_frame, f"People Detected: {num_people}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_frame, f"{range_text}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, range_color, 2)
    cv2.putText(vis_frame, f"Avg Confidence: {avg_confidence:.2f}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(vis_frame, f"Density: {overall_density:.2f} people/m²", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_frame, f"Max Local: {max_density:.2f}/m²", (20, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(vis_frame, f"Status: {status}", (20, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
    cv2.putText(vis_frame, f"Area: {area_m2:.1f} m²", (20, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Range warning indicator
    if avg_confidence < 0.6 and num_people > 0:
        warning_text = "⚠️ Possible missed detections - Check range!"
        cv2.putText(vis_frame, warning_text, (350, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    
    return vis_frame, detection_results

def process_webcam():
    """Process webcam feed with enhanced error handling and optimized performance"""
    global is_processing, current_model, frame_queue, detection_results
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        socketio.emit('error', {'message': 'Could not open webcam. Please check if webcam is connected and not being used by another application.'})
        return
    
    # Optimized webcam settings for smooth performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Balanced resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Better compression
    
    socketio.emit('webcam_started', {'message': 'Webcam started successfully'})
    
    frame_count = 0
    last_frame_time = time.time()
    
    # Optimized frame processing for smooth performance
    frame_skip = 1  # Process every frame initially
    processed_frames = 0
    last_performance_check = time.time()
    
    while is_processing:
        ret, frame = cap.read()
        if not ret:
            print("[Webcam] Failed to read frame")
            continue
        
        # Skip frames for better performance
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        try:
            # Process frame with GPU acceleration
            vis_frame, results = process_frame(frame, current_model, DEFAULT_AREA_M2, 
                                              DEFAULT_CONFIDENCE, DEFAULT_GRID_W, DEFAULT_GRID_H)
            
            # Encode frame for web streaming with optimized quality
            ret, buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                # Send to web interface
                socketio.emit('frame_update', {
                    'frame': frame_data,
                    'results': results,
                    'frame_count': frame_count
                })
            
            processed_frames += 1
            frame_count += 1
            
            # Adaptive frame skipping based on performance (less frequent checks)
            current_time = time.time()
            if current_time - last_performance_check > 2.0:  # Check every 2 seconds
                elapsed = current_time - last_frame_time
                fps = processed_frames / elapsed if elapsed > 0 else 30
                
                if fps < 15:  # If FPS is too low, skip more frames
                    frame_skip = min(4, frame_skip + 1)
                elif fps > 25:  # If FPS is good, process more frames
                    frame_skip = max(1, frame_skip - 1)
                
                last_performance_check = current_time
                last_frame_time = current_time
                processed_frames = 0
                print(f"[Webcam] Performance: FPS: {fps:.1f}, Skip: {frame_skip}, People: {results.get('people_count', 0)}")
                
        except Exception as e:
            print(f"[Webcam] Error processing frame: {e}")
            socketio.emit('error', {'message': f'Frame processing error: {str(e)}'})
            continue
        
        # Optimized timing for better performance
        current_time = time.time()
        elapsed = current_time - last_frame_time
        target_interval = 1.0 / 30.0  # 30 FPS target
        
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)
        
        last_frame_time = time.time()
    
    cap.release()

def process_video_file(video_path, area_m2, confidence):
    """Process video file with enhanced error handling"""
    global is_processing, current_model, frame_queue, detection_results
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        socketio.emit('error', {'message': 'Could not open video file. Please check if the file exists and is a valid video format.'})
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    socketio.emit('video_info', {
        'fps': fps,
        'total_frames': total_frames,
        'duration': total_frames / fps
    })
    
    frame_count = 0
    last_frame_time = time.time()
    
    # Optimized frame processing for smooth video playback
    frame_skip = 1  # Process every frame initially
    processed_frames = 0
    last_performance_check = time.time()
    
    while is_processing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[Video] End of video reached")
            break
        
        # Skip frames for better performance
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        try:
            # Process frame with GPU acceleration
            vis_frame, results = process_frame(frame, current_model, area_m2, 
                                              confidence, DEFAULT_GRID_W, DEFAULT_GRID_H)
            
            # Encode frame for web streaming with optimized quality
            ret, buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                # Send to web interface
                socketio.emit('frame_update', {
                    'frame': frame_data,
                    'results': results,
                    'progress': frame_count / total_frames if total_frames > 0 else 0,
                    'frame_count': frame_count
                })
            
            processed_frames += 1
            frame_count += 1
            
            # Adaptive frame skipping based on performance (less frequent checks)
            current_time = time.time()
            if current_time - last_performance_check > 3.0:  # Check every 3 seconds
                elapsed = current_time - last_frame_time
                fps = processed_frames / elapsed if elapsed > 0 else 30
                
                if fps < 10:  # If FPS is too low, skip more frames
                    frame_skip = min(6, frame_skip + 1)
                elif fps > 20:  # If FPS is good, process more frames
                    frame_skip = max(1, frame_skip - 1)
                
                last_performance_check = current_time
                last_frame_time = current_time
                processed_frames = 0
                print(f"[Video] Processed {frame_count}/{total_frames} frames, detected {results.get('people_count', 0)} people, FPS: {fps:.1f}, Skip: {frame_skip}")
                
        except Exception as e:
            print(f"[Video] Error processing frame {frame_count}: {e}")
            socketio.emit('error', {'message': f'Frame processing error: {str(e)}'})
            continue
        
        # Optimized timing for better performance
        current_time = time.time()
        elapsed = current_time - last_frame_time
        target_interval = 1.0 / min(fps, 30.0)  # Cap at 30 FPS max
        
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)
        
        last_frame_time = time.time()
    
    cap.release()
    socketio.emit('processing_complete', {'message': 'Video processing completed'})

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/start_webcam', methods=['POST'])
def start_webcam():
    """Start webcam processing"""
    global is_processing, processing_thread, current_model
    
    if is_processing:
        return jsonify({'error': 'Already processing'}), 400
    
    if current_model is None:
        model_path = select_best_model()
        current_model = initialize_gpu_model(model_path)
        socketio.emit('model_loaded', {
            'model': model_path,
            'device': DEVICE,
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        })
    
    is_processing = True
    processing_thread = threading.Thread(target=process_webcam)
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/api/stop_processing', methods=['POST'])
def stop_processing():
    """Stop current processing"""
    global is_processing
    
    is_processing = False
    if processing_thread:
        processing_thread.join(timeout=2)
    
    return jsonify({'status': 'stopped'})

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """Handle video file upload"""
    global is_processing, processing_thread, current_model
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    if is_processing:
        return jsonify({'error': 'Already processing'}), 400
    
    # Save uploaded file
    filename = str(uuid.uuid4()) + '_' + video_file.filename
    video_path = os.path.join(tempfile.gettempdir(), filename)
    video_file.save(video_path)
    
    # Get parameters
    area_m2 = float(request.form.get('area_m2', DEFAULT_AREA_M2))
    confidence = float(request.form.get('confidence', DEFAULT_CONFIDENCE))
    
    if current_model is None:
        model_path = select_best_model()
        current_model = initialize_gpu_model(model_path)
        socketio.emit('model_loaded', {
            'model': model_path,
            'device': DEVICE,
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        })
    
    is_processing = True
    processing_thread = threading.Thread(target=process_video_file, 
                                       args=(video_path, area_m2, confidence))
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({'status': 'started', 'filename': filename})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'message': 'Connected to stampede detection server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    global is_processing
    is_processing = False

if __name__ == '__main__':
    print("🚀 Starting Stampede Detection Web Server...")
    print("📱 Web Interface: http://localhost:5000")
    print("🎯 Model: YOLOv8 Large (High Accuracy)")
    print("📹 Features: Webcam + Video Upload")
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
