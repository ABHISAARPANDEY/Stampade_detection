#!/usr/bin/env python3
"""
Test script for Enhanced Stampede Detection System
Verifies that all components work correctly
"""

import cv2
import numpy as np
import sys
import os

def test_opencv_import():
    """Test OpenCV import and basic functionality"""
    try:
        import cv2
        print("✅ OpenCV import successful")
        
        # Test basic OpenCV functionality
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        print("✅ OpenCV basic functionality working")
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_movement_analysis():
    """Test movement analysis module"""
    try:
        from movement_analysis import MovementAnalyzer
        print("✅ Movement analysis import successful")
        
        # Test MovementAnalyzer initialization
        analyzer = MovementAnalyzer(history_size=10, flow_scale=0.5)
        print("✅ MovementAnalyzer initialization successful")
        
        # Test with dummy data
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_centers = [(100, 100), (200, 200), (300, 300)]
        test_density = np.zeros((24, 32), dtype=np.float32)
        
        # Test optical flow computation
        flow = analyzer.compute_optical_flow(test_frame, test_centers)
        print("✅ Optical flow computation working")
        
        # Test movement analysis
        result = analyzer.analyze_movement_patterns(test_frame, test_centers, test_density)
        print("✅ Movement analysis working")
        return True
    except Exception as e:
        print(f"❌ Movement analysis test failed: {e}")
        return False

def test_yolo_import():
    """Test YOLO import"""
    try:
        from ultralytics import YOLO
        print("✅ YOLO import successful")
        return True
    except Exception as e:
        print(f"❌ YOLO import failed: {e}")
        return False

def test_flask_import():
    """Test Flask import"""
    try:
        from flask import Flask
        from flask_socketio import SocketIO
        print("✅ Flask and SocketIO import successful")
        return True
    except Exception as e:
        print(f"❌ Flask import failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  GPU not available - will use CPU")
        return True
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Enhanced Stampede Detection System")
    print("=" * 50)
    
    tests = [
        ("OpenCV", test_opencv_import),
        ("Movement Analysis", test_movement_analysis),
        ("YOLO", test_yolo_import),
        ("Flask", test_flask_import),
        ("GPU", test_gpu_availability)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to run.")
        print("   Run: python start_enhanced_system_v3.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("   Install missing dependencies or fix configuration issues.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
