#!/usr/bin/env python3
"""
Enhanced STAMPede Detection System Startup Script V2
Professional Web Interface with ALL Advanced Features
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'ultralytics',
        'opencv-python', 
        'flask',
        'flask-socketio',
        'numpy',
        'torch',
        'torchvision',
        'pillow',
        'pandas',
        'scikit-learn',
        'seaborn',
        'plotly',
        'bcrypt',
        'pyjwt',
        'flask-cors',
        'flask-limiter',
        'scipy',
        'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
    
    return True

def download_yolo_model():
    """Download YOLOv8 Large model if not present"""
    model_path = Path("yolov8l.pt")
    
    if not model_path.exists():
        print("📥 Downloading YOLOv8 Large model for better accuracy...")
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8l.pt")  # This will download the model
            print("✅ YOLOv8 Large model downloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            return False
    else:
        print("✅ YOLOv8 Large model already available!")
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['templates', 'static', 'uploads', 'logs', 'reports', 'models', 'backups']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directory structure created!")

def start_enhanced_web_server():
    """Start the enhanced web server with all features"""
    print("\n🚀 Starting Enhanced STAMPede Detection System V2...")
    print("=" * 80)
    print("🎯 Model: YOLOv8 Large (GPU Accelerated)")
    print("📱 Web Interface: http://localhost:5000")
    print("🔗 API Server: http://localhost:5001")
    
    # GPU Information
    if CUDA_AVAILABLE:
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"🚀 CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  CUDA not available - using CPU")
    
    print("\n📹 Enhanced Features:")
    print("   ✅ Multi-camera support (up to 8 cameras)")
    print("   ✅ SQLite database with full analytics")
    print("   ✅ Real-time alerts with sound notifications")
    print("   ✅ Mobile responsive design")
    print("   ✅ Heat map visualization")
    print("   ✅ Predictive analytics & trend analysis")
    print("   ✅ RESTful API for integrations")
    print("   ✅ User authentication & role-based access")
    print("   ✅ Historical data analysis & reporting")
    print("   ✅ Advanced crowd flow analysis")
    print("   ✅ Risk assessment & movement detection")
    print("   ✅ Automated report generation")
    print("   ✅ Webhook notifications")
    print("   ✅ Data export capabilities")
    print("   ✅ 3D visualization support")
    print("   ✅ Custom model training ready")
    print("=" * 80)
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(3)
        webbrowser.open('http://localhost:5000')
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the enhanced web server
    try:
        from enhanced_web_server import app, socketio
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Enhanced STAMPede Detection System...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

def start_api_server():
    """Start the API server in a separate process"""
    try:
        import subprocess
        api_process = subprocess.Popen([
            sys.executable, 'api_server.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("🔗 API Server started on port 5001")
        return api_process
    except Exception as e:
        print(f"⚠️  Failed to start API server: {e}")
        return None

def main():
    """Main startup function with all enhanced features"""
    print("🛡️  Enhanced STAMPede Detection System V2")
    print("Professional Web Interface with ALL Advanced Features")
    print("=" * 80)
    print("🔧 Complete Feature Set:")
    print("   • Multi-camera monitoring (up to 8 cameras)")
    print("   • SQLite database with full data persistence")
    print("   • Real-time alerts with sound & notifications")
    print("   • Mobile responsive design for all devices")
    print("   • Advanced heat map visualization")
    print("   • Predictive analytics & machine learning")
    print("   • RESTful API for third-party integrations")
    print("   • User authentication & role-based access control")
    print("   • Historical data analysis & reporting")
    print("   • Advanced crowd flow & movement analysis")
    print("   • Risk assessment & stampede prediction")
    print("   • Automated report generation (PDF/HTML/JSON)")
    print("   • Webhook notifications & email alerts")
    print("   • Data export capabilities (CSV/JSON)")
    print("   • 3D visualization & AR overlay ready")
    print("   • Custom model training & transfer learning")
    print("   • Shift management & operational features")
    print("   • Security features & audit logging")
    print("=" * 80)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required!")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Download model
    if not download_yolo_model():
        print("⚠️  Continuing with available model...")
    
    print("\n🎉 All enhancements are ready!")
    print("   The system now includes ALL advanced features:")
    print("   • Multi-camera support with load balancing")
    print("   • Complete database integration with analytics")
    print("   • Real-time alerts with multiple notification channels")
    print("   • Mobile responsive design for all screen sizes")
    print("   • Advanced heat map visualization with multiple styles")
    print("   • Predictive analytics with machine learning models")
    print("   • RESTful API with comprehensive endpoints")
    print("   • User authentication with role-based permissions")
    print("   • Historical data analysis with trend detection")
    print("   • Advanced crowd flow and movement analysis")
    print("   • Risk assessment with multi-factor analysis")
    print("   • Automated report generation with charts")
    print("   • Webhook notifications and email alerts")
    print("   • Data export in multiple formats")
    print("   • 3D visualization and AR overlay capabilities")
    print("   • Custom model training and transfer learning")
    print("   • Shift management and operational features")
    print("   • Security features with audit logging")
    
    # GPU Status
    if CUDA_AVAILABLE:
        print(f"\n🔥 GPU Acceleration: ENABLED ({torch.cuda.get_device_name(0)})")
        print("   • Smooth multi-camera performance")
        print("   • Fast video processing")
        print("   • Real-time detection with all features")
        print("   • Advanced analytics processing")
    else:
        print("\n⚠️  GPU Acceleration: DISABLED (CPU mode)")
        print("   • Install CUDA-enabled PyTorch for better performance")
        print("   • All features work on CPU but may be slower")
    
    # Start API server in background
    api_process = start_api_server()
    
    # Start enhanced web server
    if not start_enhanced_web_server():
        sys.exit(1)
    
    # Cleanup
    if api_process:
        api_process.terminate()

if __name__ == "__main__":
    main()
