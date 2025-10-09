"""
RESTful API Server for STAMPede Detection System
Provides comprehensive API endpoints for third-party integrations
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import threading
import os
from functools import wraps

# Import our modules
from database_manager import DatabaseManager, DetectionRecord, AlertRecord
from alert_manager import AlertManager, AlertLevel, AlertType
from multi_camera_manager import MultiCameraManager, CameraConfig
from predictive_analytics import CrowdPredictor

app = Flask(__name__)
CORS(app)

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per hour"]
)

# Initialize managers
db_manager = DatabaseManager()
alert_manager = AlertManager()
camera_manager = MultiCameraManager()
predictor = CrowdPredictor()

# API version
API_VERSION = "v1"

def require_auth(f):
    """Simple authentication decorator (can be enhanced)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Simple API key authentication
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.getenv('STAMPEDE_API_KEY', 'default_key'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

def validate_json_data(required_fields: List[str]):
    """Validate JSON request data"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({'error': 'Request must be JSON'}), 400
            
            data = request.get_json()
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return jsonify({
                    'error': f'Missing required fields: {missing_fields}'
                }), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Health Check
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': API_VERSION,
        'services': {
            'database': 'connected',
            'alerts': 'active',
            'cameras': 'active',
            'predictions': 'active'
        }
    })

# Detection Data Endpoints
@app.route(f'/api/{API_VERSION}/detections', methods=['GET'])
@limiter.limit("100 per minute")
def get_detections():
    """Get detection records with optional filters"""
    try:
        # Parse query parameters
        camera_id = request.args.get('camera_id', type=int)
        start_time = request.args.get('start_time', type=float)
        end_time = request.args.get('end_time', type=float)
        limit = request.args.get('limit', 1000, type=int)
        
        # Get records from database
        records = db_manager.get_detection_records(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        # Convert to JSON-serializable format
        detections = []
        for record in records:
            detections.append({
                'id': record.id,
                'timestamp': record.timestamp,
                'camera_id': record.camera_id,
                'people_count': record.people_count,
                'density': record.density,
                'max_density': record.max_density,
                'avg_density': record.avg_density,
                'status': record.status,
                'alert_level': record.alert_level,
                'risk_score': record.risk_score,
                'risk_level': record.risk_level,
                'flow_intensity': record.flow_intensity,
                'movement_direction': record.movement_direction,
                'movement_risk_score': record.movement_risk_score,
                'movement_risk_level': record.movement_risk_level,
                'detection_boxes': json.loads(record.detection_boxes),
                'confidence_scores': json.loads(record.confidence_scores),
                'risk_factors': json.loads(record.risk_factors),
                'movement_risk_factors': json.loads(record.movement_risk_factors),
                'area_m2': record.area_m2,
                'confidence_threshold': record.confidence_threshold,
                'grid_w': record.grid_w,
                'grid_h': record.grid_h
            })
        
        return jsonify({
            'detections': detections,
            'count': len(detections),
            'timestamp': time.time()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(f'/api/{API_VERSION}/detections', methods=['POST'])
@require_auth
@validate_json_data(['camera_id', 'people_count', 'density'])
def create_detection():
    """Create a new detection record"""
    try:
        data = request.get_json()
        
        # Create detection record
        record = DetectionRecord(
            timestamp=data.get('timestamp', time.time()),
            camera_id=data['camera_id'],
            people_count=data['people_count'],
            density=data['density'],
            max_density=data.get('max_density', data['density']),
            avg_density=data.get('avg_density', data['density']),
            status=data.get('status', 'SAFE'),
            alert_level=data.get('alert_level', 'safe'),
            risk_score=data.get('risk_score', 0.0),
            risk_level=data.get('risk_level', 'low'),
            flow_intensity=data.get('flow_intensity', 0.0),
            movement_direction=data.get('movement_direction', 'stable'),
            movement_risk_score=data.get('movement_risk_score', 0.0),
            movement_risk_level=data.get('movement_risk_level', 'low'),
            detection_boxes=json.dumps(data.get('detection_boxes', [])),
            confidence_scores=json.dumps(data.get('confidence_scores', [])),
            risk_factors=json.dumps(data.get('risk_factors', [])),
            movement_risk_factors=json.dumps(data.get('movement_risk_factors', [])),
            area_m2=data.get('area_m2', 25.0),
            confidence_threshold=data.get('confidence_threshold', 0.20),
            grid_w=data.get('grid_w', 32),
            grid_h=data.get('grid_h', 24)
        )
        
        # Insert into database
        record_id = db_manager.insert_detection_record(record)
        
        # Add to predictor for analytics
        predictor.add_data_point(
            camera_id=record.camera_id,
            people_count=record.people_count,
            density=record.density,
            flow_intensity=record.flow_intensity,
            movement_risk_score=record.movement_risk_score,
            timestamp=record.timestamp
        )
        
        return jsonify({
            'id': record_id,
            'message': 'Detection record created successfully',
            'timestamp': time.time()
        }), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Alert Endpoints
@app.route(f'/api/{API_VERSION}/alerts', methods=['GET'])
@limiter.limit("50 per minute")
def get_alerts():
    """Get alert records"""
    try:
        camera_id = request.args.get('camera_id', type=int)
        acknowledged = request.args.get('acknowledged', type=bool)
        start_time = request.args.get('start_time', type=float)
        end_time = request.args.get('end_time', type=float)
        limit = request.args.get('limit', 100, type=int)
        
        records = db_manager.get_alert_records(
            camera_id=camera_id,
            acknowledged=acknowledged,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        alerts = []
        for record in records:
            alerts.append({
                'id': record.id,
                'timestamp': record.timestamp,
                'camera_id': record.camera_id,
                'alert_type': record.alert_type,
                'alert_level': record.alert_level,
                'message': record.message,
                'people_count': record.people_count,
                'density': record.density,
                'risk_score': record.risk_score,
                'acknowledged': record.acknowledged,
                'acknowledged_by': record.acknowledged_by,
                'acknowledged_at': record.acknowledged_at
            })
        
        return jsonify({
            'alerts': alerts,
            'count': len(alerts),
            'timestamp': time.time()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(f'/api/{API_VERSION}/alerts/<int:alert_id>/acknowledge', methods=['POST'])
@require_auth
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    try:
        data = request.get_json()
        acknowledged_by = data.get('acknowledged_by', 'api_user')
        
        success = db_manager.acknowledge_alert(alert_id, acknowledged_by)
        
        if success:
            return jsonify({
                'message': 'Alert acknowledged successfully',
                'timestamp': time.time()
            })
        else:
            return jsonify({'error': 'Alert not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Analytics Endpoints
@app.route(f'/api/{API_VERSION}/analytics/summary', methods=['GET'])
@limiter.limit("20 per minute")
def get_analytics_summary():
    """Get analytics summary"""
    try:
        camera_id = request.args.get('camera_id', type=int)
        start_time = request.args.get('start_time', type=float)
        end_time = request.args.get('end_time', type=float)
        
        summary = db_manager.get_analytics_summary(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time
        )
        
        return jsonify({
            'summary': summary,
            'timestamp': time.time()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(f'/api/{API_VERSION}/analytics/predictions/<int:camera_id>', methods=['GET'])
@limiter.limit("10 per minute")
def get_predictions(camera_id):
    """Get predictions for a camera"""
    try:
        predictions = predictor.get_predictions_summary(camera_id)
        return jsonify(predictions)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Camera Management Endpoints
@app.route(f'/api/{API_VERSION}/cameras', methods=['GET'])
def get_cameras():
    """Get camera configurations"""
    try:
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
                'error_count': error_count
            })
        
        return jsonify({
            'cameras': cameras,
            'count': len(cameras),
            'timestamp': time.time()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(f'/api/{API_VERSION}/cameras', methods=['POST'])
@require_auth
@validate_json_data(['camera_id', 'name'])
def add_camera():
    """Add a new camera"""
    try:
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
            return jsonify({
                'message': 'Camera added successfully',
                'camera_id': config.camera_id,
                'timestamp': time.time()
            }), 201
        else:
            return jsonify({'error': 'Failed to add camera'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(f'/api/{API_VERSION}/cameras/<int:camera_id>/start', methods=['POST'])
@require_auth
def start_camera(camera_id):
    """Start a camera"""
    try:
        success = camera_manager.start_camera(camera_id)
        
        if success:
            return jsonify({
                'message': f'Camera {camera_id} started successfully',
                'timestamp': time.time()
            })
        else:
            return jsonify({'error': f'Failed to start camera {camera_id}'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(f'/api/{API_VERSION}/cameras/<int:camera_id>/stop', methods=['POST'])
@require_auth
def stop_camera(camera_id):
    """Stop a camera"""
    try:
        success = camera_manager.stop_camera(camera_id)
        
        if success:
            return jsonify({
                'message': f'Camera {camera_id} stopped successfully',
                'timestamp': time.time()
            })
        else:
            return jsonify({'error': f'Failed to stop camera {camera_id}'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# System Status Endpoints
@app.route(f'/api/{API_VERSION}/status', methods=['GET'])
def get_system_status():
    """Get system status"""
    try:
        # Get database stats
        db_stats = db_manager.get_database_stats()
        
        # Get camera status
        camera_status = camera_manager.get_all_camera_status()
        
        # Get alert stats
        alert_stats = alert_manager.get_alert_stats()
        
        return jsonify({
            'system': {
                'status': 'running',
                'uptime': time.time(),
                'version': API_VERSION
            },
            'database': db_stats,
            'cameras': {
                'total': len(camera_manager.cameras),
                'active': len([s for s in camera_status.values() if s.value == 'connected']),
                'status': {str(k): v.value for k, v in camera_status.items()}
            },
            'alerts': alert_stats,
            'timestamp': time.time()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Webhook Endpoints
@app.route(f'/api/{API_VERSION}/webhooks', methods=['POST'])
@require_auth
@validate_json_data(['url'])
def add_webhook():
    """Add a webhook URL"""
    try:
        data = request.get_json()
        url = data['url']
        headers = data.get('headers', {})
        
        alert_manager.webhook_manager.add_webhook(url, headers)
        
        return jsonify({
            'message': 'Webhook added successfully',
            'url': url,
            'timestamp': time.time()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Data Export Endpoints
@app.route(f'/api/{API_VERSION}/export/detections', methods=['GET'])
@require_auth
def export_detections():
    """Export detection data as CSV"""
    try:
        camera_id = request.args.get('camera_id', type=int)
        start_time = request.args.get('start_time', type=float)
        end_time = request.args.get('end_time', type=float)
        
        # Get data
        records = db_manager.get_detection_records(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        # Convert to CSV
        csv_data = "timestamp,camera_id,people_count,density,status,risk_score\n"
        for record in records:
            csv_data += f"{record.timestamp},{record.camera_id},{record.people_count},{record.density},{record.status},{record.risk_score}\n"
        
        return Response(
            csv_data,
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=detections.csv'}
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({'error': 'Rate limit exceeded'}), 429

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting STAMPede Detection API Server...")
    print("📡 API Documentation: http://localhost:5001/api/health")
    print("🔑 API Key: Set STAMPEDE_API_KEY environment variable")
    
    # Start camera manager
    camera_manager.start()
    
    # Start alert manager
    alert_manager.start()
    
    # Run API server
    app.run(host='0.0.0.0', port=5001, debug=False)
