# STAMPede Detection System

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the system
python start_enhanced_system.py
```

## 📋 System Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ RAM
- Webcam or video files for testing

## 🎯 Features

- **Real-time Detection**: YOLOv8 Large model with GPU acceleration
- **Professional Interface**: Clean web dashboard with live metrics
- **Advanced Analytics**: Multi-factor risk assessment and crowd flow analysis
- **Simple Visualization**: Clear dots instead of cluttered bounding boxes
- **Smart Alerts**: Conservative thresholds prevent false alarms

## 📁 Project Structure

```
person-detection/
├── web_server.py              # Main web application
├── stampede.py                # Core detection algorithm  
├── start_enhanced_system.py   # System startup script
├── train.py                   # Model training script
├── templates/
│   └── index.html            # Web interface
├── requirements.txt           # Dependencies
└── FINAL_PROJECT_DOCUMENTATION.md  # Complete documentation
```

## 🔧 Configuration

The system automatically detects and uses GPU acceleration if available. Key parameters:

- **Confidence**: 0.15 (optimized for dense crowds)
- **Image Size**: 1280px (high resolution)
- **Grid Resolution**: 32x24 (fine analysis)
- **Risk Thresholds**: Conservative to prevent false alarms

## 📖 Documentation

See `FINAL_PROJECT_DOCUMENTATION.md` for complete technical details, academic Q&A, and implementation specifics.

## 🎓 Academic Use

This project demonstrates:
- Computer vision applications
- Real-time processing systems
- GPU acceleration techniques
- Web application development
- Risk assessment algorithms

Perfect for computer science, engineering, and AI/ML courses.
