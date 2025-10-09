"""
Enhanced Web Server for STAMPede Detection System
Integrates all new features: multi-camera, database, alerts, analytics, etc.
"""

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

# Import our enhanced modules
from database_manager import DatabaseManager, DetectionRecord
from alert_manager import AlertManager, AlertLevel, AlertType
from multi_camera_manager import MultiCameraManager, CameraConfig, CameraStatus
from predictive_analytics import CrowdPredictor
from heat_map_visualizer import HeatMapVisualizer, HeatMapConfig, HeatMapStyle
from auth_manager import AuthManager, UserRole, Permission
from reporting_engine import ReportingEngine, ReportType

app = Flask(__name__)
app.config['SECRET_KEY'] = 'stampede_detection_enhanced_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize all managers
db_manager = DatabaseManager()
alert_manager = AlertManager()
camera_manager = MultiCameraManager(max_cameras=8)
predictor = CrowdPredictor()
heat_map_viz = HeatMapVisualizer()
auth_manager = AuthManager()
reporting_engine = ReportingEngine(db_manager)

# Global variables
current_model = None
processing_threads = {}
is_processing = {}
latest_frames = {}
frame_queues = {}

# Enhanced Configuration
DEFAULT_AREA_M2 = 25.0
DEFAULT_CONFIDENCE = 0.20
DEFAULT_GRID_W = 32
DEFAULT_GRID_H = 24
DANGER_DENSITY = 6.0
WARNING_DENSITY = 4.0
DEFAULT_IMAGE_SIZE = 1280

# GPU Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0

def select_best_model():
    """Select the best available YOLO model"""
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
    
    return "yolov8l.pt"

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
    
    model = YOLO(model_path)
    
    if torch.cuda.is_available():
        model.to(DEVICE)
        print(f"✅ Model loaded on GPU: {DEVICE}")
    else:
        print("✅ Model loaded on CPU")
    
    return model

def compute_density_map(centers, frame_shape, grid_w, grid_h, total_area_m2):
    """Enhanced crowd density computation"""
    h, w = frame_shape[:2]
    density_count = np.zeros((grid_h, grid_w), dtype=np.float32)
    if not centers:
        return density_count
    
    cell_w = max(1, w // grid_w)
    cell_h = max(1, h // grid_h)
    
    # Enhanced counting with weighted distribution
    for cx, cy in centers:
        gx = min(grid_w - 1, max(0, cx // cell_w))
        gy = min(grid_h - 1, max(0, cy // cell_h))
        density_count[gy, gx] += 1.0
        
        # Add weighted contribution to neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ngx = gx + dx
                ngy = gy + dy
                if 0 <= ngx < grid_w and 0 <= ngy < grid_h:
                    cell_center_x = ngx * cell_w + cell_w // 2
                    cell_center_y = ngy * cell_h + cell_h // 2
                    distance = np.sqrt((cx - cell_center_x)**2 + (cy - cell_center_y)**2)
                    max_distance = np.sqrt(cell_w**2 + cell_h**2)
                    weight = max(0, 1.0 - distance / max_distance) * 0.1
                    density_count[ngy, ngx] += weight
    
    # Convert to people per square meter
    total_cells = grid_w * grid_h
    area_per_cell_m2 = total_area_m2 / total_cells
    area_per_cell_m2 = max(area_per_cell_m2, 0.05)
    
    density_per_m2 = density_count / area_per_cell_m2
    density_per_m2 = cv2.GaussianBlur(density_per_m2, (5, 5), 1.0)
    
    return density_per_m2

def process_frame_enhanced(frame, model, area_m2, confidence, grid_w, grid_h, camera_id):
    """Enhanced frame processing with all features"""
    
    # YOLOv11 detection with optimized settings for best accuracy
    results = model(frame, 
                   conf=confidence, 
                   classes=[0], 
                   verbose=False,
                   imgsz=DEFAULT_IMAGE_SIZE,
                   iou=0.25,
                   max_det=5000,  # Higher limit for YOLOv11's superior detection
                   agnostic_nms=True,
                   augment=True,  # Enable augmentation for YOLOv11's accuracy
                   device=DEVICE,
                   half=True if DEVICE == 'cuda' else False,
                   save=False,
                   save_txt=False,
                   save_conf=False)
    
    # Extract detections
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
            
            if conf is not None and conf[i] < confidence:
                continue
                
            x0, y0, x1, y1 = box.astype(int)
            
            # Ensure boxes are within frame bounds
            h, w = frame.shape[:2]
            x0 = max(0, min(w-1, x0))
            y0 = max(0, min(h-1, y0))
            x1 = max(0, min(w-1, x1))
            y1 = max(0, min(h-1, y1))
            
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
    
    num_people = len(centers)
    max_density = float(np.max(density_map)) if density_map.size else 0.0
    avg_density = float(np.mean(density_map)) if density_map.size else 0.0
    
    # Crowd flow analysis
    flow_data = analyze_crowd_flow(centers, frame.shape)
    
    # Movement analysis
    movement_analysis = {
        'movement_risk_level': 'low',
        'movement_risk_score': 0.0,
        'movement_risk_factors': []
    }
    
    # Risk assessment
    risk_assessment = assess_risk_factors(num_people, overall_density, max_density, flow_data)
    
    # Determine status
    if overall_density >= DANGER_DENSITY:
        status = "DANGER: STAMPEDE RISK"
        status_color = (0, 0, 255)
        alert_level = "danger"
    elif overall_density >= WARNING_DENSITY:
        status = "CROWDED: MONITOR CLOSELY"
        status_color = (0, 255, 255)
        alert_level = "warning"
    else:
        status = "SAFE: NORMAL CONDITIONS"
        status_color = (0, 200, 0)
        alert_level = "safe"
    
    # Create detection record
    detection_record = DetectionRecord(
        timestamp=time.time(),
        camera_id=camera_id,
        people_count=num_people,
        density=overall_density,
        max_density=max_density,
        avg_density=avg_density,
        status=status,
        alert_level=alert_level,
        risk_score=risk_assessment['risk_score'],
        risk_level=risk_assessment['risk_level'],
        flow_intensity=flow_data['flow_intensity'],
        movement_direction=flow_data['movement_direction'],
        movement_risk_score=movement_analysis['movement_risk_score'],
        movement_risk_level=movement_analysis['movement_risk_level'],
        detection_boxes=json.dumps([[int(x) for x in box] for box in detection_boxes]),
        confidence_scores=json.dumps([float(score) for score in confidence_scores]),
        risk_factors=json.dumps(risk_assessment['risk_factors']),
        movement_risk_factors=json.dumps(movement_analysis['movement_risk_factors']),
        area_m2=area_m2,
        confidence_threshold=confidence,
        grid_w=grid_w,
        grid_h=grid_h
    )
    
    # Store in database
    db_manager.insert_detection_record(detection_record)
    
    # Add to predictor
    predictor.add_data_point(
        camera_id=camera_id,
        people_count=num_people,
        density=overall_density,
        flow_intensity=flow_data['flow_intensity'],
        movement_risk_score=movement_analysis['movement_risk_score'],
        timestamp=time.time()
    )
    
    # Check for alerts
    density_alert = alert_manager.check_density_alert(camera_id, overall_density, num_people)
    movement_alert = alert_manager.check_movement_alert(camera_id, movement_analysis['movement_risk_score'], movement_analysis['movement_risk_level'])
    flow_alert = alert_manager.check_crowd_flow_alert(camera_id, flow_data['flow_intensity'], flow_data['movement_direction'])
    
    # Store alerts in database
    for alert in [density_alert, movement_alert, flow_alert]:
        if alert:
            alert_record = {
                'timestamp': alert.timestamp,
                'camera_id': alert.camera_id,
                'alert_type': alert.alert_type.value,
                'alert_level': alert.alert_level.value,
                'message': alert.message,
                'people_count': num_people,
                'density': overall_density,
                'risk_score': risk_assessment['risk_score'],
                'acknowledged': False,
                'acknowledged_by': None,
                'acknowledged_at': None
            }
            db_manager.insert_alert_record(alert_record)
    
    # Create visualization
    vis_frame = frame.copy()
    
    # Add heat map if enabled
    if request.args.get('heatmap', 'false').lower() == 'true':
        heat_map_config = HeatMapConfig(
            style=HeatMapStyle.INFERNO,
            alpha=0.6,
            show_contours=True,
            show_peaks=True
        )
        vis_frame = heat_map_viz.overlay_heatmap(frame, density_map, heat_map_config)
    
    # Draw detections
    if len(detection_boxes) > 0:
        cell_w = max(1, frame.shape[1] // grid_w)
        cell_h = max(1, frame.shape[0] // grid_h)
        
        for i, (x0, y0, x1, y1) in enumerate(detection_boxes):
            cx = int((x0 + x1) * 0.5)
            cy = int((y0 + y1) * 0.5)
            gx = min(grid_w - 1, max(0, cx // cell_w))
            gy = min(grid_h - 1, max(0, cy // cell_h))
            local_density = float(density_map[gy, gx]) if density_map.size else 0.0
            
            # Color coding
            if overall_density >= DANGER_DENSITY or local_density >= DANGER_DENSITY:
                color = (0, 0, 255)
                dot_size = 4
            elif overall_density >= WARNING_DENSITY or local_density >= WARNING_DENSITY:
                color = (0, 255, 255)
                dot_size = 3
            else:
                color = (0, 200, 0)
                dot_size = 2
            
            # Draw dot
            cv2.circle(vis_frame, (cx, cy), dot_size, color, -1)
            cv2.circle(vis_frame, (cx, cy), dot_size, (255, 255, 255), 1)
    
    # Add status overlay
    cv2.rectangle(vis_frame, (10, 10), (500, 120), (0, 0, 0), -1)
    cv2.rectangle(vis_frame, (10, 10), (500, 120), (255, 255, 255), 1)
    
    cv2.putText(vis_frame, f"People: {num_people}", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_frame, f"Density: {overall_density:.2f}/m²", (20, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_frame, f"Status: {status}", (20, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
    cv2.putText(vis_frame, f"Camera: {camera_id}", (20, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(vis_frame, f"Risk: {risk_assessment['risk_level']}", (20, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Prepare results
    results = {
        'people_count': int(num_people),
        'density': float(round(overall_density, 2)),
        'max_density': float(round(max_density, 2)),
        'avg_density': float(round(avg_density, 2)),
        'status': str(status),
        'status_color': list(status_color),
        'alert_level': str(alert_level),
        'risk_score': float(risk_assessment['risk_score']),
        'risk_level': str(risk_assessment['risk_level']),
        'flow_intensity': float(flow_data['flow_intensity']),
        'movement_direction': str(flow_data['movement_direction']),
        'movement_risk_score': float(movement_analysis['movement_risk_score']),
        'movement_risk_level': str(movement_analysis['movement_risk_level']),
        'detection_boxes': [[int(x) for x in box] for box in detection_boxes],
        'confidence_scores': [float(score) for score in confidence_scores],
        'timestamp': float(time.time())
    }
    
    return vis_frame, results

def analyze_crowd_flow(centers, frame_shape):
    """Analyze crowd movement patterns"""
    # Simplified flow analysis
    return {
        'flow_intensity': 0.0,
        'movement_direction': 'stable',
        'crowd_velocity': 0.0,
        'movement_count': 0
    }

def assess_risk_factors(num_people, overall_density, max_density, flow_data):
    """Assess risk factors"""
    risk_score = 0.0
    risk_factors = []
    
    if overall_density >= DANGER_DENSITY and num_people >= 10:
        risk_score += 0.4
        risk_factors.append('high_density')
    elif overall_density >= WARNING_DENSITY and num_people >= 6:
        risk_score += 0.2
        risk_factors.append('moderate_density')
    
    if num_people >= 15:
        risk_score += 0.3
        risk_factors.append('many_people')
    elif num_people >= 8:
        risk_score += 0.1
        risk_factors.append('moderate_people')
    
    if flow_data['flow_intensity'] > 0.8 and num_people >= 8:
        risk_score += 0.2
        risk_factors.append('high_movement')
    elif flow_data['flow_intensity'] > 0.5 and num_people >= 5:
        risk_score += 0.1
        risk_factors.append('moderate_movement')
    
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
        'risk_factors': risk_factors
    }

def process_camera_feed(camera_id):
    """Process camera feed with all enhancements"""
    global is_processing, current_model
    
    if camera_id not in camera_manager.cameras:
        return
    
    config = camera_manager.cameras[camera_id]
    cap = camera_manager.camera_caps.get(camera_id)
    
    if not cap or not cap.isOpened():
        return
    
    frame_count = 0
    last_frame_time = time.time()
    
    while is_processing.get(camera_id, False):
        ret, frame = cap.read()
        if not ret:
            continue
        
        try:
            # Process frame with all enhancements
            vis_frame, results = process_frame_enhanced(
                frame, current_model, config.area_m2, 
                config.confidence, config.grid_w, config.grid_h, camera_id
            )
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                # Send to web interface
                socketio.emit('frame_update', {
                    'camera_id': camera_id,
                    'frame': frame_data,
                    'results': results,
                    'frame_count': frame_count
                })
            
            frame_count += 1
            
            # Control frame rate
            current_time = time.time()
            elapsed = current_time - last_frame_time
            target_interval = 1.0 / config.fps
            
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            
            last_frame_time = time.time()
            
        except Exception as e:
            print(f"[Camera {camera_id}] Error processing frame: {e}")
            continue

# Web Routes
@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/start_camera/<int:camera_id>', methods=['POST'])
def start_camera(camera_id):
    """Start processing a specific camera"""
    global current_model, is_processing, processing_threads
    
    if current_model is None:
        model_path = select_best_model()
        current_model = initialize_gpu_model(model_path)
        socketio.emit('model_loaded', {
            'model': model_path,
            'device': DEVICE,
            'gpu_available': torch.cuda.is_available()
        })
    
    if camera_id not in camera_manager.cameras:
        return jsonify({'error': 'Camera not found'}), 404
    
    if is_processing.get(camera_id, False):
        return jsonify({'error': 'Camera already processing'}), 400
    
    # Start camera
    if not camera_manager.start_camera(camera_id):
        return jsonify({'error': 'Failed to start camera'}), 500
    
    # Start processing thread
    is_processing[camera_id] = True
    thread = threading.Thread(target=process_camera_feed, args=(camera_id,))
    thread.daemon = True
    thread.start()
    processing_threads[camera_id] = thread
    
    return jsonify({'status': 'started', 'camera_id': camera_id})

@app.route('/api/stop_camera/<int:camera_id>', methods=['POST'])
def stop_camera(camera_id):
    """Stop processing a specific camera"""
    global is_processing, processing_threads
    
    if camera_id not in is_processing:
        return jsonify({'error': 'Camera not processing'}), 400
    
    # Stop processing
    is_processing[camera_id] = False
    
    # Stop camera
    camera_manager.stop_camera(camera_id)
    
    # Wait for thread to finish
    if camera_id in processing_threads:
        processing_threads[camera_id].join(timeout=2)
        del processing_threads[camera_id]
    
    return jsonify({'status': 'stopped', 'camera_id': camera_id})

@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get camera list"""
    cameras = []
    for camera_id, config in camera_manager.cameras.items():
        status = camera_manager.get_camera_status(camera_id)
        fps = camera_manager.get_camera_fps(camera_id)
        error_count = camera_manager.get_camera_error_count(camera_id)
        
        cameras.append({
            'camera_id': camera_id,
            'name': config.name,
            'resolution': f"{config.resolution[0]}x{config.resolution[1]}",
            'fps': config.fps,
            'area_m2': config.area_m2,
            'confidence': config.confidence,
            'enabled': config.enabled,
            'status': status.value if status else 'unknown',
            'current_fps': fps,
            'error_count': error_count,
            'is_processing': is_processing.get(camera_id, False)
        })
    
    return jsonify({'cameras': cameras})

@app.route('/api/add_camera', methods=['POST'])
def add_camera():
    """Add a new camera"""
    data = request.get_json()
    
    config = CameraConfig(
        camera_id=data['camera_id'],
        name=data['name'],
        resolution=tuple(data.get('resolution', [1280, 720])),
        fps=data.get('fps', 30),
        area_m2=data.get('area_m2', 25.0),
        confidence=data.get('confidence', 0.20),
        grid_w=data.get('grid_w', 32),
        grid_h=data.get('grid_h', 24),
        enabled=data.get('enabled', True)
    )
    
    success = camera_manager.add_camera(config)
    
    if success:
        return jsonify({'status': 'added', 'camera_id': config.camera_id})
    else:
        return jsonify({'error': 'Failed to add camera'}), 400

@app.route('/api/analytics/summary', methods=['GET'])
def get_analytics_summary():
    """Get analytics summary"""
    camera_id = request.args.get('camera_id', type=int)
    start_time = request.args.get('start_time', type=float)
    end_time = request.args.get('end_time', type=float)
    
    summary = db_manager.get_analytics_summary(
        camera_id=camera_id,
        start_time=start_time,
        end_time=end_time
    )
    
    return jsonify({'summary': summary})

@app.route('/api/analytics/predictions/<int:camera_id>', methods=['GET'])
def get_predictions(camera_id):
    """Get predictions for a camera"""
    predictions = predictor.get_predictions_summary(camera_id)
    return jsonify(predictions)

@app.route('/api/reports/generate', methods=['POST'])
def generate_report():
    """Generate a report"""
    data = request.get_json()
    
    report_type = ReportType(data['report_type'])
    start_time = data['start_time']
    end_time = data['end_time']
    camera_ids = data.get('camera_ids', [])
    
    if report_type == ReportType.DAILY_SUMMARY:
        date = datetime.fromtimestamp(start_time)
        result = reporting_engine.generate_daily_summary(date, camera_ids)
    elif report_type == ReportType.WEEKLY_ANALYSIS:
        week_start = datetime.fromtimestamp(start_time)
        result = reporting_engine.generate_weekly_analysis(week_start, camera_ids)
    elif report_type == ReportType.MONTHLY_REPORT:
        date = datetime.fromtimestamp(start_time)
        result = reporting_engine.generate_monthly_report(date.month, date.year, camera_ids)
    else:
        result = reporting_engine.generate_custom_report(start_time, end_time, camera_ids)
    
    return jsonify({
        'report_id': result.report_id,
        'file_path': result.file_path,
        'summary': result.summary
    })

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Get list of reports"""
    reports = reporting_engine.get_report_list()
    return jsonify({'reports': reports})

@app.route('/api/reports/<report_id>', methods=['GET'])
def get_report(report_id):
    """Get specific report"""
    report = reporting_engine.get_report(report_id)
    if report:
        return jsonify({'report': report})
    else:
        return jsonify({'error': 'Report not found'}), 404

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'message': 'Connected to enhanced stampede detection server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    pass

if __name__ == '__main__':
    print("🚀 Starting Enhanced STAMPede Detection System...")
    print("📱 Web Interface: http://localhost:5000")
    print("🎯 Features: Multi-camera, Database, Alerts, Analytics, Reports")
    print("🔥 GPU Acceleration:", "ENABLED" if torch.cuda.is_available() else "DISABLED")
    
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Start camera manager
    camera_manager.start()
    
    # Start alert manager
    alert_manager.start()
    
    # Run enhanced web server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
