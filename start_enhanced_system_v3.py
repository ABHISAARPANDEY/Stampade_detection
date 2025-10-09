#!/usr/bin/env python3
"""
Enhanced Stampede Detection System Startup Script v3
Optimized for Smooth Performance and Error-Free Operation
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
    directories = ['templates', 'static', 'uploads', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directory structure created!")

def start_web_server():
    """Start the enhanced web server with GPU acceleration"""
    print("\n🚀 Starting Enhanced Stampede Detection System v3...")
    print("=" * 70)
    print("🎯 Model: YOLOv8 Large (GPU Accelerated)")
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
    print("   • Enhanced dense crowd detection (confidence: 0.20)")
    print("   • Optimized resolution processing (1280px max)")
    print("   • GPU acceleration for smooth performance")
    print("   • Smart alert system with cooldown")
    print("   • Real-time density mapping and trends")
    print("   • Adaptive frame skipping for performance")
    print("   • Fixed OpenCV optical flow errors")
    print("   • Optimized video processing pipeline")
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
    print("🛡️  Enhanced Stampede Detection System v3")
    print("Optimized for Smooth Performance and Error-Free Operation")
    print("=" * 70)
    print("🔧 Recent Optimizations:")
    print("   • Fixed OpenCV optical flow errors")
    print("   • Optimized video processing pipeline")
    print("   • Added adaptive frame skipping")
    print("   • Improved GPU utilization")
    print("   • Reduced memory usage")
    print("   • Enhanced error handling")
    print("   • Better performance monitoring")
    print("   • Smoother video playback")
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
    
    print("\n🎉 All optimizations are ready!")
    print("   The system now includes:")
    print("   • Fixed OpenCV optical flow errors")
    print("   • Optimized video processing pipeline")
    print("   • Adaptive frame skipping for performance")
    print("   • Better GPU utilization")
    print("   • Smoother video playback")
    print("   • Enhanced error handling")
    print("   • Reduced memory usage")
    
    # GPU Status
    if CUDA_AVAILABLE:
        print(f"🔥 GPU Acceleration: ENABLED ({torch.cuda.get_device_name(0)})")
        print("   • Smooth webcam performance")
        print("   • Fast video processing")
        print("   • Real-time detection")
        print("   • Optimized memory usage")
    else:
        print("⚠️  GPU Acceleration: DISABLED (CPU mode)")
        print("   • Install CUDA-enabled PyTorch for better performance")
    
    # Start web server
    if not start_web_server():
        sys.exit(1)

if __name__ == "__main__":
    main()
