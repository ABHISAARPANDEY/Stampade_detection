"""
Multi-Camera Manager for STAMPede Detection System
Handles multiple camera feeds simultaneously with load balancing
"""

import cv2
import threading
import time
import queue
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque

class CameraStatus(Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"

@dataclass
class CameraConfig:
    camera_id: int
    name: str
    resolution: tuple = (1280, 720)
    fps: int = 30
    enabled: bool = True
    area_m2: float = 25.0
    confidence: float = 0.20
    grid_w: int = 32
    grid_h: int = 24

@dataclass
class CameraFrame:
    camera_id: int
    frame: np.ndarray
    timestamp: float
    frame_number: int
    detection_results: Optional[dict] = None

class MultiCameraManager:
    """Manages multiple camera feeds with load balancing and failover"""
    
    def __init__(self, max_cameras: int = 4):
        self.max_cameras = max_cameras
        self.cameras: Dict[int, CameraConfig] = {}
        self.camera_threads: Dict[int, threading.Thread] = {}
        self.camera_caps: Dict[int, cv2.VideoCapture] = {}
        self.camera_status: Dict[int, CameraStatus] = {}
        self.frame_queues: Dict[int, queue.Queue] = {}
        self.is_running = False
        self.frame_callbacks: List[Callable] = []
        self.detection_callbacks: List[Callable] = []
        
        # Performance monitoring
        self.fps_counters: Dict[int, deque] = {}
        self.error_counts: Dict[int, int] = {}
        
    def add_camera(self, config: CameraConfig) -> bool:
        """Add a new camera configuration"""
        if len(self.cameras) >= self.max_cameras:
            print(f"[MultiCamera] Maximum cameras ({self.max_cameras}) reached")
            return False
        
        if config.camera_id in self.cameras:
            print(f"[MultiCamera] Camera {config.camera_id} already exists")
            return False
        
        self.cameras[config.camera_id] = config
        self.camera_status[config.camera_id] = CameraStatus.DISCONNECTED
        self.frame_queues[config.camera_id] = queue.Queue(maxsize=10)
        self.fps_counters[config.camera_id] = deque(maxlen=30)
        self.error_counts[config.camera_id] = 0
        
        print(f"[MultiCamera] Added camera {config.camera_id}: {config.name}")
        return True
    
    def remove_camera(self, camera_id: int) -> bool:
        """Remove a camera and stop its processing"""
        if camera_id not in self.cameras:
            return False
        
        # Stop camera thread
        if camera_id in self.camera_threads:
            self.camera_threads[camera_id].join(timeout=2)
            del self.camera_threads[camera_id]
        
        # Release camera
        if camera_id in self.camera_caps:
            self.camera_caps[camera_id].release()
            del self.camera_caps[camera_id]
        
        # Clean up data structures
        del self.cameras[camera_id]
        del self.camera_status[camera_id]
        del self.frame_queues[camera_id]
        del self.fps_counters[camera_id]
        del self.error_counts[camera_id]
        
        print(f"[MultiCamera] Removed camera {camera_id}")
        return True
    
    def start_camera(self, camera_id: int) -> bool:
        """Start processing a specific camera"""
        if camera_id not in self.cameras:
            return False
        
        if camera_id in self.camera_threads and self.camera_threads[camera_id].is_alive():
            print(f"[MultiCamera] Camera {camera_id} already running")
            return True
        
        config = self.cameras[camera_id]
        if not config.enabled:
            print(f"[MultiCamera] Camera {camera_id} is disabled")
            return False
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"[MultiCamera] Failed to open camera {camera_id}")
            self.camera_status[camera_id] = CameraStatus.ERROR
            return False
        
        # Configure camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, config.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.camera_caps[camera_id] = cap
        self.camera_status[camera_id] = CameraStatus.CONNECTED
        
        # Start processing thread
        thread = threading.Thread(target=self._process_camera, args=(camera_id,))
        thread.daemon = True
        thread.start()
        self.camera_threads[camera_id] = thread
        
        print(f"[MultiCamera] Started camera {camera_id}: {config.name}")
        return True
    
    def stop_camera(self, camera_id: int) -> bool:
        """Stop processing a specific camera"""
        if camera_id not in self.cameras:
            return False
        
        # Release camera
        if camera_id in self.camera_caps:
            self.camera_caps[camera_id].release()
            del self.camera_caps[camera_id]
        
        self.camera_status[camera_id] = CameraStatus.DISCONNECTED
        
        # Wait for thread to finish
        if camera_id in self.camera_threads:
            self.camera_threads[camera_id].join(timeout=2)
            del self.camera_threads[camera_id]
        
        print(f"[MultiCamera] Stopped camera {camera_id}")
        return True
    
    def start_all_cameras(self) -> bool:
        """Start all enabled cameras"""
        success_count = 0
        for camera_id in self.cameras:
            if self.cameras[camera_id].enabled:
                if self.start_camera(camera_id):
                    success_count += 1
        
        print(f"[MultiCamera] Started {success_count}/{len(self.cameras)} cameras")
        return success_count > 0
    
    def stop_all_cameras(self) -> bool:
        """Stop all cameras"""
        for camera_id in list(self.cameras.keys()):
            self.stop_camera(camera_id)
        
        print("[MultiCamera] Stopped all cameras")
        return True
    
    def _process_camera(self, camera_id: int):
        """Process frames from a specific camera"""
        cap = self.camera_caps[camera_id]
        config = self.cameras[camera_id]
        frame_count = 0
        last_time = time.time()
        
        while self.is_running and camera_id in self.camera_caps:
            ret, frame = cap.read()
            if not ret:
                print(f"[MultiCamera] Camera {camera_id} failed to read frame")
                self.error_counts[camera_id] += 1
                time.sleep(0.1)
                continue
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time) if current_time > last_time else 0
            self.fps_counters[camera_id].append(fps)
            last_time = current_time
            
            # Create frame object
            camera_frame = CameraFrame(
                camera_id=camera_id,
                frame=frame.copy(),
                timestamp=current_time,
                frame_number=frame_count
            )
            
            # Add to queue (non-blocking)
            try:
                self.frame_queues[camera_id].put_nowait(camera_frame)
            except queue.Full:
                # Remove oldest frame if queue is full
                try:
                    self.frame_queues[camera_id].get_nowait()
                    self.frame_queues[camera_id].put_nowait(camera_frame)
                except queue.Empty:
                    pass
            
            # Notify callbacks
            for callback in self.frame_callbacks:
                try:
                    callback(camera_frame)
                except Exception as e:
                    print(f"[MultiCamera] Frame callback error: {e}")
            
            frame_count += 1
            
            # Control frame rate
            target_interval = 1.0 / config.fps
            elapsed = time.time() - current_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
        
        print(f"[MultiCamera] Camera {camera_id} processing stopped")
    
    def get_latest_frame(self, camera_id: int) -> Optional[CameraFrame]:
        """Get the latest frame from a specific camera"""
        if camera_id not in self.frame_queues:
            return None
        
        try:
            return self.frame_queues[camera_id].get_nowait()
        except queue.Empty:
            return None
    
    def get_all_latest_frames(self) -> Dict[int, CameraFrame]:
        """Get latest frames from all cameras"""
        frames = {}
        for camera_id in self.cameras:
            frame = self.get_latest_frame(camera_id)
            if frame:
                frames[camera_id] = frame
        return frames
    
    def add_frame_callback(self, callback: Callable[[CameraFrame], None]):
        """Add a callback for new frames"""
        self.frame_callbacks.append(callback)
    
    def add_detection_callback(self, callback: Callable[[int, dict], None]):
        """Add a callback for detection results"""
        self.detection_callbacks.append(callback)
    
    def get_camera_status(self, camera_id: int) -> Optional[CameraStatus]:
        """Get status of a specific camera"""
        return self.camera_status.get(camera_id)
    
    def get_all_camera_status(self) -> Dict[int, CameraStatus]:
        """Get status of all cameras"""
        return self.camera_status.copy()
    
    def get_camera_fps(self, camera_id: int) -> float:
        """Get average FPS for a camera"""
        if camera_id not in self.fps_counters or not self.fps_counters[camera_id]:
            return 0.0
        return sum(self.fps_counters[camera_id]) / len(self.fps_counters[camera_id])
    
    def get_camera_error_count(self, camera_id: int) -> int:
        """Get error count for a camera"""
        return self.error_counts.get(camera_id, 0)
    
    def start(self):
        """Start the multi-camera manager"""
        self.is_running = True
        self.start_all_cameras()
        print("[MultiCamera] Manager started")
    
    def stop(self):
        """Stop the multi-camera manager"""
        self.is_running = False
        self.stop_all_cameras()
        print("[MultiCamera] Manager stopped")
    
    def get_camera_config(self, camera_id: int) -> Optional[CameraConfig]:
        """Get configuration for a specific camera"""
        return self.cameras.get(camera_id)
    
    def update_camera_config(self, camera_id: int, **kwargs) -> bool:
        """Update configuration for a specific camera"""
        if camera_id not in self.cameras:
            return False
        
        config = self.cameras[camera_id]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        print(f"[MultiCamera] Updated camera {camera_id} configuration")
        return True
