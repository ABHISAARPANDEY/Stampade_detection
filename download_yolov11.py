#!/usr/bin/env python3
"""
YOLOv11 Model Downloader
Downloads the latest YOLOv11 Large model for best accuracy
"""

import os
import sys
from pathlib import Path

def download_yolov11_large():
    """Download YOLOv11 Large model for best accuracy."""
    print("🚀 Downloading YOLOv11 Large Model for Best Accuracy")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics library found")
        
        # Download YOLOv11 Large model
        print("📥 Downloading YOLOv11 Large model...")
        model = YOLO("yolov11l.pt")
        
        # Verify the model was downloaded
        if os.path.exists("yolov11l.pt"):
            file_size = os.path.getsize("yolov11l.pt") / (1024 * 1024)  # MB
            print(f"✅ YOLOv11 Large model downloaded successfully!")
            print(f"📊 File size: {file_size:.1f} MB")
            print(f"📁 Location: {os.path.abspath('yolov11l.pt')}")
            
            # Test the model
            print("\n🧪 Testing model...")
            results = model("https://ultralytics.com/images/bus.jpg", verbose=False)
            print("✅ Model test successful!")
            
            return True
        else:
            print("❌ Model download failed")
            return False
            
    except ImportError:
        print("❌ Ultralytics library not found")
        print("📦 Installing ultralytics...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            print("✅ Ultralytics installed successfully")
            return download_yolov11_large()  # Retry after installation
        except Exception as e:
            print(f"❌ Failed to install ultralytics: {e}")
            return False
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def main():
    """Main function."""
    success = download_yolov11_large()
    
    if success:
        print("\n🎉 YOLOv11 Large model is ready for accuracy testing!")
        print("📋 Next steps:")
        print("   1. Run: python run_accuracy_tests.py")
        print("   2. Or run: python model_accuracy_evaluator.py")
    else:
        print("\n❌ Failed to download YOLOv11 Large model")
        print("📋 Alternative options:")
        print("   1. Manually download from: https://github.com/ultralytics/assets/releases")
        print("   2. Use existing YOLOv8 model if available")
        print("   3. Check internet connection and try again")

if __name__ == "__main__":
    main()
