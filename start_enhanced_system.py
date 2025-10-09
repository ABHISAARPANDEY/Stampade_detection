#!/usr/bin/env python3
"""
Enhanced Stampede Detection System Startup Script
Professional Web Interface with YOLOv11 Large Model for Best Accuracy
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
        'pillow'
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
    """Download YOLOv11 Large model if not present for best accuracy"""
    model_path = Path("yolov11l.pt")
    
    if not model_path.exists():
        print("📥 Downloading YOLOv11 Large model for best accuracy...")
        try:
            from ultralytics import YOLO
            model = YOLO("yolov11l.pt")  # This will download the model
            print("✅ YOLOv11 Large model downloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to download YOLOv11 Large: {e}")
            print("🔄 Falling back to YOLOv8 Large...")
            try:
                model = YOLO("yolov8l.pt")
                print("✅ YOLOv8 Large model downloaded as fallback!")
            except Exception as e2:
                print(f"❌ Failed to download any model: {e2}")
                return False
    else:
        print("✅ YOLOv11 Large model already available!")
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['templates', 'static', 'uploads', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directory structure created!")

def start_web_server():
    """Start the enhanced web server with GPU acceleration"""
    print("\n🚀 Starting Enhanced Stampede Detection System...")
    print("=" * 70)
    print("🎯 Model: YOLOv11 Large (GPU Accelerated) - Best Accuracy")
    print("📱 Web Interface: http://localhost:5000")
    
    # GPU Information
    if CUDA_AVAILABLE:
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"🚀 CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  CUDA not available - using CPU")
    
    print("📹 Enhanced Features:")
    print("   • Real-time webcam detection with clear dots")
    print("   • Video file upload and processing")
    print("   • Professional dashboard with detailed metrics")
    print("   • Advanced crowd flow analysis")
    print("   • Multi-factor risk assessment")
    print("   • Enhanced dense crowd detection (confidence: 0.15)")
    print("   • Higher resolution processing (1280px)")
    print("   • GPU acceleration for smooth performance")
    print("   • Smart alert system with cooldown")
    print("   • Real-time density mapping and trends")
    print("=" * 70)
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:5000')
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the web server
    try:
        from web_server import app, socketio
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Enhanced Stampede Detection System...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

def main():
    """Main startup function with enhanced features"""
    print("🛡️  Enhanced Stampede Detection System")
    print("Professional Web Interface with Advanced Dense Crowd Detection")
    print("=" * 70)
    print("🔧 Recent Enhancements:")
    print("   • Fixed analysis cropping issues")
    print("   • Enhanced dense crowd detection")
    print("   • Added bounding box visualization")
    print("   • Improved density calculation algorithms")
    print("   • Advanced risk assessment system")
    print("   • Crowd flow analysis capabilities")
    print("   • Smart alert system with cooldown")
    print("=" * 70)
    
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
    print("   The system now includes:")
    print("   • Better dense crowd detection (confidence: 0.15)")
    print("   • Higher resolution processing (1280px)")
    print("   • Clear simple dot visualization")
    print("   • Advanced risk assessment and crowd flow analysis")
    print("   • Fixed cropping issues for full frame analysis")
    
    # GPU Status
    if CUDA_AVAILABLE:
        print(f"🔥 GPU Acceleration: ENABLED ({torch.cuda.get_device_name(0)})")
        print("   • Smooth webcam performance")
        print("   • Fast video processing")
        print("   • Real-time detection")
    else:
        print("⚠️  GPU Acceleration: DISABLED (CPU mode)")
        print("   • Install CUDA-enabled PyTorch for better performance")
    
    # Start web server
    if not start_web_server():
        sys.exit(1)

if __name__ == "__main__":
    main()
