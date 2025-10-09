import argparse
import os
from typing import List, Tuple, Dict
import threading
from collections import deque
import time
import json

import cv2
import numpy as np
from ultralytics import YOLO
from movement_analysis import MovementAnalyzer

# Shared buffer for live streaming
latest_jpeg = deque(maxlen=1)
flask_app = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stampede risk detection with YOLOv8 tracking")
    parser.add_argument("--video", type=str, default="demo_video2.mp4", help="Path to input video file")
    parser.add_argument("--webcam", type=int, default=None, help="Webcam device ID (0, 1, 2...) - overrides --video")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to YOLOv11 weights (.pt). If not provided, tries ./training/... then falls back to bundled yolov11l.pt",
    )
    parser.add_argument("--conf", type=float, default=0.15, help="YOLOv11 confidence threshold (optimized for best accuracy)")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO image size (increased for better accuracy)")
    parser.add_argument("--out", type=str, default="stampede_output.mp4", help="Path to save annotated video")
    parser.add_argument("--grid_w", type=int, default=32, help="Number of grid cells horizontally (increased for finer analysis)")
    parser.add_argument("--grid_h", type=int, default=24, help="Number of grid cells vertically (increased for finer analysis)")
    parser.add_argument("--area_m2", type=float, default=None, help="Total monitored area in square meters (REQUIRED)")
    parser.add_argument("--test_mode", action="store_true", help="Enable test mode with detailed density logging")
    parser.add_argument("--danger_density", type=float, default=6.0, help="Danger threshold (people/m²) - red alert (stampede/crush risk) - user requirement: 6 people/m²")
    parser.add_argument("--warning_density", type=float, default=4.0, help="Warning threshold (people/m²) - yellow alert (crowded but manageable) - user requirement: 4 people/m²")
    parser.add_argument("--smooth_frames", type=int, default=90, help="Frames to smooth density over (~3 sec at 30fps for faster response)")
    parser.add_argument("--display", action="store_true", help="Show live preview window")
    parser.add_argument("--fullscreen", action="store_true", help="Show display window in fullscreen mode")
    parser.add_argument("--serve", action="store_true", help="Serve live preview at http://HOST:PORT/")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Live server host")
    parser.add_argument("--port", type=int, default=5000, help="Live server port")
    parser.add_argument("--device", type=str, default="", help="Inference device, e.g. '0' for CUDA:0 or 'cpu'")
    parser.add_argument("--skip_frames", type=int, default=0, help="Skip processing every N frames (0=no skip)")
    parser.add_argument("--flow_scale", type=float, default=0.5, help="Downscale factor for optical flow (0.25-1.0)")
    parser.add_argument("--flow_every", type=int, default=1, help="Compute optical flow every N frames (>=1)")
    parser.add_argument("--jpeg_quality", type=int, default=70, help="JPEG quality for live stream (10-95)")
    parser.add_argument("--max_stream_fps", type=int, default=20, help="Max FPS for live stream updates")
    parser.add_argument("--motion", type=str, choices=["tracks", "flow"], default="tracks", help="Motion backend: track velocities (fast) or optical flow (heavier)")
    parser.add_argument("--iou_threshold", type=float, default=0.25, help="IoU threshold for NMS (lower for better dense crowd detection)")
    parser.add_argument("--max_detections", type=int, default=3000, help="Maximum number of detections per frame")
    parser.add_argument("--agnostic_nms", action="store_true", help="Use class-agnostic NMS for better dense crowd handling")
    parser.add_argument("--crowd_flow", action="store_true", help="Enable crowd flow analysis for movement detection")
    parser.add_argument("--risk_assessment", action="store_true", help="Enable advanced risk assessment with multiple factors")
    parser.add_argument("--alert_cooldown", type=int, default=30, help="Alert cooldown period in seconds to prevent spam")
    parser.add_argument("--movement_analysis", action="store_true", help="Enable advanced movement analysis (involuntary flow, bottleneck, panic, wave motion)")
    parser.add_argument("--movement_sensitivity", type=float, default=0.5, help="Movement analysis sensitivity (0.1-1.0)")
    return parser.parse_args()


def select_weights(user_weights: str | None) -> str:
    if user_weights and os.path.exists(user_weights):
        return user_weights
    # Prioritize YOLOv11 Large for best accuracy, fallback to YOLOv8
    candidates: List[str] = [
        "./yolo11l.pt",  # YOLOv11 Large - best accuracy for dense crowds
        "./yolov11l.pt",  # Alternative naming
        "./yolov11m.pt",  # YOLOv11 Medium
        "./yolov11s.pt",  # YOLOv11 Small
        "./yolov11n.pt",  # YOLOv11 Nano
        "./training/yolo11l/train/weights/best.pt",  # Custom trained YOLOv11 Large
        "./training/yolov11l/train/weights/best.pt",
        "./training/yolov11m/train/weights/best.pt",
        "./training/yolov11s/train/weights/best.pt",
        "./training/yolov11n/train/weights/best.pt",
        "./training/yolov8l/train/weights/best.pt",  # Fallback to YOLOv8
        "./training/yolov8m/train/weights/best.pt",
        "./training/yolov8s/train/weights/best.pt",
        "./training/yolov8n/train/weights/best.pt",
        "./yolov8l.pt",  # YOLOv8 Large fallback
        "./yolov8m.pt",
        "./yolov8s.pt",
        "./yolov8n.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Default to YOLOv11 Large for best real-world accuracy and dense crowd detection
    return "yolo11l.pt"


def compute_density_map(
    centers: List[Tuple[int, int]], frame_shape: Tuple[int, int, int], grid_w: int, grid_h: int, total_area_m2: float
) -> np.ndarray:
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
        # This helps with people on grid boundaries
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
    
    # Ensure minimum area per cell to prevent extreme densities
    area_per_cell_m2 = max(area_per_cell_m2, 0.05)  # Reduced minimum for better sensitivity
    
    density_per_m2 = density_count / area_per_cell_m2
    
    # Enhanced spatial smoothing for better density estimation
    density_per_m2 = cv2.GaussianBlur(density_per_m2, (5, 5), 1.0)  # Increased smoothing
    
    return density_per_m2


def compute_motion_map(
    flow: np.ndarray | None, grid_w: int, grid_h: int
) -> np.ndarray:
    if flow is None:
        return np.zeros((grid_h, grid_w), dtype=np.float32)
    h, w = flow.shape[:2]
    mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag = cv2.GaussianBlur(mag, (3, 3), 0)
    resized = cv2.resize(mag, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def smooth_density_temporal(
    current_density: np.ndarray, density_history: List[np.ndarray], max_frames: int
) -> np.ndarray:
    """Apply temporal smoothing to reduce frame-by-frame noise in density measurements."""
    density_history.append(current_density.copy())
    
    # Keep only recent frames
    if len(density_history) > max_frames:
        density_history.pop(0)
    
    # Average over recent frames
    if len(density_history) == 1:
        return current_density
    
    stacked = np.stack(density_history, axis=0)
    smoothed = np.mean(stacked, axis=0)
    return smoothed


def analyze_crowd_flow(centers: List[Tuple[int, int]], frame_shape: Tuple[int, int, int]) -> Dict:
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
    
    # Simple movement analysis (can be enhanced with optical flow)
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
    
    # Determine movement direction (simplified)
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


def assess_risk_factors(num_people: int, overall_density: float, max_density: float, 
                       flow_data: Dict, args) -> Dict:
    """Advanced risk assessment using multiple factors."""
    global risk_factors
    
    # Update trend data
    risk_factors['density_trend'].append(overall_density)
    risk_factors['people_trend'].append(num_people)
    risk_factors['movement_intensity'].append(flow_data['flow_intensity'])
    
    # Calculate trends
    density_trend = 0.0
    people_trend = 0.0
    movement_trend = 0.0
    
    if len(risk_factors['density_trend']) >= 3:
        recent_density = list(risk_factors['density_trend'])[-3:]
        density_trend = (recent_density[-1] - recent_density[0]) / max(recent_density[0], 0.1)
    
    if len(risk_factors['people_trend']) >= 3:
        recent_people = list(risk_factors['people_trend'])[-3:]
        people_trend = (recent_people[-1] - recent_people[0]) / max(recent_people[0], 1)
    
    if len(risk_factors['movement_intensity']) >= 3:
        recent_movement = list(risk_factors['movement_intensity'])[-3:]
        movement_trend = recent_movement[-1] - recent_movement[0]
    
    # Risk scoring
    risk_score = 0.0
    risk_factors_list = []
    
    # Density factor
    if overall_density >= args.danger_density:
        risk_score += 0.4
        risk_factors_list.append('high_density')
    elif overall_density >= args.warning_density:
        risk_score += 0.2
        risk_factors_list.append('moderate_density')
    
    # People count factor
    if num_people >= 10:
        risk_score += 0.3
        risk_factors_list.append('many_people')
    elif num_people >= 5:
        risk_score += 0.1
        risk_factors_list.append('moderate_people')
    
    # Movement factor
    if flow_data['flow_intensity'] > 0.7:
        risk_score += 0.2
        risk_factors_list.append('high_movement')
    elif flow_data['flow_intensity'] > 0.3:
        risk_score += 0.1
        risk_factors_list.append('moderate_movement')
    
    # Trend factors
    if density_trend > 0.5:  # Rapid density increase
        risk_score += 0.1
        risk_factors_list.append('increasing_density')
    
    if people_trend > 0.3:  # Rapid people increase
        risk_score += 0.1
        risk_factors_list.append('increasing_crowd')
    
    # Determine overall risk level
    if risk_score >= 0.7:
        risk_level = 'critical'
    elif risk_score >= 0.5:
        risk_level = 'high'
    elif risk_score >= 0.3:
        risk_level = 'moderate'
    else:
        risk_level = 'low'
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_factors': risk_factors_list,
        'density_trend': density_trend,
        'people_trend': people_trend,
        'movement_trend': movement_trend
    }


def overlay_heatmap(base_bgr: np.ndarray, density_map: np.ndarray, max_density: float = 10.0) -> np.ndarray:
    """Overlay density heatmap where intensity represents people per square meter."""
    h, w = base_bgr.shape[:2]
    density_resized = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_LINEAR)
    # Normalize density to 0-255 range (cap at max_density for visualization)
    density_normalized = np.clip(density_resized / max_density, 0, 1)
    density_uint8 = (density_normalized * 255).astype(np.uint8)
    heat = cv2.applyColorMap(density_uint8, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(base_bgr, 0.7, heat, 0.3, 0)
    return overlay


def draw_density_alerts(
    frame: np.ndarray, density_map: np.ndarray, warning_thresh: float, danger_thresh: float
) -> None:
    """Draw grid alerts based on crowd density thresholds."""
    h, w = frame.shape[:2]
    gh, gw = density_map.shape[:2]
    cell_w = w // gw
    cell_h = h // gh
    
    for gy in range(gh):
        for gx in range(gw):
            density = float(density_map[gy, gx])
            x0 = gx * cell_w
            y0 = gy * cell_h
            x1 = w if gx == gw - 1 else (gx + 1) * cell_w
            y1 = h if gy == gh - 1 else (gy + 1) * cell_h
            
            if density >= danger_thresh:
                # Red for danger zone (6+ people/m²)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 3)
                cv2.putText(frame, f"{density:.1f}/m²", (x0+5, y0+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif density >= warning_thresh:
                # Yellow for warning (4+ people/m²)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.putText(frame, f"{density:.1f}/m²", (x0+5, y0+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


# Global variables for calibration and advanced features
calibration_points = []
calibration_mode = ""
calibration_complete = False

# Advanced features global variables
last_alert_time = 0
crowd_flow_history = deque(maxlen=30)  # Store recent movement data
risk_factors = {
    'density_trend': deque(maxlen=10),
    'people_trend': deque(maxlen=10),
    'movement_intensity': deque(maxlen=10)
}

# Movement analyzer instance
movement_analyzer = None


def calibration_mouse_callback(event, x, y, flags, param):
    """Mouse callback for area calibration."""
    global calibration_points, calibration_mode, calibration_complete
    
    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_points.append((x, y))
        print(f"[Calibration] Point {len(calibration_points)}: ({x}, {y})")
        
        if calibration_mode == "rectangle" and len(calibration_points) == 2:
            calibration_complete = True
        elif calibration_mode == "reference" and len(calibration_points) == 2:
            calibration_complete = True


def calibrate_area_interactive(cap, method="rectangle"):
    """Interactive area calibration using mouse clicks."""
    global calibration_points, calibration_mode, calibration_complete
    
    calibration_points = []
    calibration_mode = method
    calibration_complete = False
    
    print(f"\n[Calibration] Starting {method} calibration...")
    
    if method == "rectangle":
        print("Click two opposite corners of a known rectangular area")
        print("Press ESC to cancel, ENTER when done")
    elif method == "reference":
        print("Click two ends of a reference object of known size")
        print("Press ESC to cancel, ENTER when done")
    
    cv2.namedWindow("Area Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Area Calibration", calibration_mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Calibration] Failed to read frame")
            return None
            
        # Draw existing points
        for i, point in enumerate(calibration_points):
            cv2.circle(frame, point, 8, (0, 255, 0), -1)
            cv2.putText(frame, f"P{i+1}", (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw line between points if we have 2
        if len(calibration_points) == 2:
            cv2.line(frame, calibration_points[0], calibration_points[1], (0, 255, 0), 2)
            
            # Calculate pixel distance
            p1, p2 = calibration_points
            pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            if method == "rectangle":
                cv2.putText(frame, f"Rectangle diagonal: {pixel_distance:.1f} pixels", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Enter the real diagonal length in meters:", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            elif method == "reference":
                cv2.putText(frame, f"Reference length: {pixel_distance:.1f} pixels", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "This will be used to calculate scale", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        if len(calibration_points) < 2:
            if method == "rectangle":
                cv2.putText(frame, f"Click corner {len(calibration_points)+1} of rectangular area", 
                           (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            elif method == "reference":
                cv2.putText(frame, f"Click end {len(calibration_points)+1} of reference object", 
                           (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Press ENTER to confirm, ESC to cancel, R to restart", 
                       (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Area Calibration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("[Calibration] Cancelled")
            cv2.destroyWindow("Area Calibration")
            return None
        elif key == 13 and len(calibration_points) == 2:  # ENTER
            break
        elif key == ord('r'):  # Restart
            calibration_points = []
            print("[Calibration] Restarting...")
    
    cv2.destroyWindow("Area Calibration")
    
    if len(calibration_points) == 2:
        p1, p2 = calibration_points
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return {
            'points': calibration_points,
            'pixel_distance': pixel_distance,
            'method': method
        }
    
    return None


def calculate_area_from_calibration(calibration_data, frame_shape, real_measurement):
    """Calculate total monitored area from calibration data."""
    if not calibration_data:
        return None
    
    pixel_distance = calibration_data['pixel_distance']
    method = calibration_data['method']
    h, w = frame_shape[:2]
    
    # Calculate pixels per meter
    pixels_per_meter = pixel_distance / real_measurement
    
    if method == "rectangle":
        # User measured diagonal of a rectangle
        # Estimate total frame area
        frame_diagonal_pixels = np.sqrt(w**2 + h**2)
        frame_diagonal_meters = frame_diagonal_pixels / pixels_per_meter
        
        # Rough approximation: assume frame shows rectangular area
        # with aspect ratio matching frame aspect ratio
        aspect_ratio = w / h
        frame_height_meters = frame_diagonal_meters / np.sqrt(1 + aspect_ratio**2)
        frame_width_meters = frame_height_meters * aspect_ratio
        total_area_m2 = frame_width_meters * frame_height_meters
        
    elif method == "reference":
        # User measured a reference object
        # Calculate total frame area
        total_pixels = w * h
        pixels_per_m2 = pixels_per_meter ** 2
        total_area_m2 = total_pixels / pixels_per_m2
    
    else:
        return None
    
    return {
        'area_m2': total_area_m2,
        'pixels_per_meter': pixels_per_meter,
        'method': method,
        'measurement': real_measurement
    }


def save_calibration_data(calibration_result, filename):
    """Save calibration data to JSON file for reuse."""
    try:
        with open(filename, 'w') as f:
            json.dump(calibration_result, f, indent=2)
        print(f"[Calibration] Saved to {filename}")
        return True
    except Exception as e:
        print(f"[Calibration] Failed to save: {e}")
        return False


def load_calibration_data(filename):
    """Load calibration data from JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"[Calibration] Loaded from {filename}: {data['area_m2']:.1f} m²")
        return data
    except Exception as e:
        print(f"[Calibration] Failed to load {filename}: {e}")
        return None


def validate_calibration(area_m2, frame_shape):
    """Validate calibration makes sense."""
    h, w = frame_shape[:2]
    frame_pixels = h * w
    
    # Sanity checks
    if area_m2 < 0.1:
        print(f"[WARNING] Area too small: {area_m2:.1f} m². Minimum recommended: 0.5 m²")
        return False
    elif area_m2 > 1000:
        print(f"[WARNING] Area very large: {area_m2:.1f} m². Are you sure?")
        confirm = input("Continue anyway? (y/n): ").lower().strip()
        return confirm == 'y'
    
    # Check if reasonable pixels per m²
    pixels_per_m2 = frame_pixels / area_m2
    if pixels_per_m2 < 1000:
        print(f"[WARNING] Very low resolution per m²: {pixels_per_m2:.0f} pixels/m²")
        print("This may affect detection accuracy.")
    
    return True


def run_area_calibration(cap, frame_shape):
    """Run the enhanced area calibration process."""
    print("\n" + "="*60)
    print("REAL-WORLD AREA CALIBRATION")
    print("="*60)
    print("For accurate stampede detection, we need the exact monitored area.")
    print("\nRecommended methods:")
    print("1. Rectangle method - Click corners of known rectangular area")
    print("   (Best for: rooms, corridors, defined spaces)")
    print("2. Reference object - Click ends of object with known size") 
    print("   (Best for: when you have a ruler, door, table, etc.)")
    print("3. Manual input - Enter area directly")
    print("   (Best for: when you've pre-measured the space)")
    
    while True:
        choice = input("\nEnter choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    result = None
    
    if choice == '3':
        # Manual input with validation
        print("\nMANUAL AREA INPUT")
        print("Measure your monitoring area with a tape measure.")
        print("For rectangular areas: length × width")
        print("For irregular areas: estimate total square meters")
        
        while True:
            try:
                area = float(input("Enter total monitored area in square meters: "))
                if area > 0 and validate_calibration(area, frame_shape):
                    result = {
                        'area_m2': area,
                        'method': 'manual',
                        'pixels_per_meter': None,
                        'timestamp': time.time()
                    }
                    break
                else:
                    print("Please enter a valid positive area.")
            except ValueError:
                print("Please enter a valid number.")
    
    elif choice == '1':
        # Rectangle method with better instructions
        print("\nRECTANGLE CALIBRATION")
        print("1. Identify a rectangular area in your camera view")
        print("2. Measure its diagonal with a tape measure")
        print("3. Click the two diagonal corners on screen")
        
        calibration_data = calibrate_area_interactive(cap, "rectangle")
        if calibration_data:
            while True:
                try:
                    diagonal_meters = float(input("Enter the diagonal length in meters: "))
                    if diagonal_meters > 0:
                        break
                    else:
                        print("Length must be positive.")
                except ValueError:
                    print("Please enter a valid number.")
            
            result = calculate_area_from_calibration(calibration_data, frame_shape, diagonal_meters)
            if result and validate_calibration(result['area_m2'], frame_shape):
                result['timestamp'] = time.time()
                print(f"[Calibration] Calculated area: {result['area_m2']:.1f} m²")
                print(f"[Calibration] Scale: {result['pixels_per_meter']:.1f} pixels/meter")
            else:
                result = None
    
    elif choice == '2':
        # Reference object method with better instructions
        print("\nREFERENCE OBJECT CALIBRATION")
        print("1. Place an object of known size in the camera view")
        print("2. Good objects: ruler, door width (≈0.8m), table length")
        print("3. Click both ends of the object on screen")
        
        calibration_data = calibrate_area_interactive(cap, "reference")
        if calibration_data:
            while True:
                try:
                    object_size = float(input("Enter the reference object size in meters: "))
                    if object_size > 0:
                        break
                    else:
                        print("Size must be positive.")
                except ValueError:
                    print("Please enter a valid number.")
            
            result = calculate_area_from_calibration(calibration_data, frame_shape, object_size)
            if result and validate_calibration(result['area_m2'], frame_shape):
                result['timestamp'] = time.time()
                print(f"[Calibration] Calculated area: {result['area_m2']:.1f} m²")
                print(f"[Calibration] Scale: {result['pixels_per_meter']:.1f} pixels/meter")
            else:
                result = None
    
    if result:
        # Show test scenarios
        print(f"\n" + "="*40)
        print("CALIBRATION COMPLETE")
        print(f"Monitored area: {result['area_m2']:.1f} m²")
        print(f"="*40)
        print("Expected density readings for testing:")
        print(f"• 1 person: {1/result['area_m2']:.1f} people/m² (should be GREEN)")
        print(f"• 4 people: {4/result['area_m2']:.1f} people/m² (should be {'GREEN' if 4/result['area_m2'] < 3 else 'YELLOW' if 4/result['area_m2'] < 5 else 'ORANGE' if 4/result['area_m2'] < 6 else 'RED'})")
        print(f"• 8 people: {8/result['area_m2']:.1f} people/m² (should be {'GREEN' if 8/result['area_m2'] < 3 else 'YELLOW' if 8/result['area_m2'] < 5 else 'ORANGE' if 8/result['area_m2'] < 6 else 'RED'})")
        return result
    else:
        print("[Calibration] Calibration failed or cancelled.")
        return None


def main() -> None:
    args = parse_args()
    weights = select_weights(args.weights)

    # Determine video source
    if args.webcam is not None:
        video_source = args.webcam
        print(f"[Stampede] Using weights: {weights}")
        print(f"[Stampede] Opening webcam: {args.webcam}")
    else:
        video_source = args.video
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Input video not found: {args.video}")
        print(f"[Stampede] Using weights: {weights}")
        print(f"[Stampede] Opening video: {args.video}")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        source_desc = f"webcam {args.webcam}" if args.webcam is not None else f"video {args.video}"
        raise RuntimeError(f"Failed to open {source_desc}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # For webcams, frame count is often 0 or unreliable
    if args.webcam is not None:
        print(f"[Stampede] Webcam info: {width}x{height} @ {fps:.2f} FPS (live stream)")
        total_frames = 0  # Disable progress bar for live streams
    else:
        print(f"[Stampede] Video info: {width}x{height} @ {fps:.2f} FPS, frames={total_frames}")

    # Simple area input - user provides exact area
    if args.area_m2 is None:
        print("\n" + "="*50)
        print("AREA INPUT REQUIRED")
        print("="*50)
        print("Enter the total monitored area in square meters.")
        print("Examples:")
        print("• Small room: 10-20 m²")
        print("• Large room: 30-50 m²") 
        print("• Corridor: 15-25 m²")
        print("• Open area: 50+ m²")
        
        while True:
            try:
                area_input = input("\nEnter monitored area (m²): ").strip()
                monitored_area = float(area_input)
                if monitored_area > 0:
                    break
                else:
                    print("Area must be positive.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        monitored_area = args.area_m2
    
    print(f"[Stampede] Using area: {monitored_area:.1f} m²")
    
    # Show expected thresholds for this area (user-specified system)
    print(f"[Stampede] Density thresholds for {monitored_area:.1f} m²:")
    print(f"  • GREEN (Safe): < {int(monitored_area * 4)} people (< 4.0 people/m²) - Safe conditions")
    print(f"  • YELLOW (Crowded): {int(monitored_area * 4)}-{int(monitored_area * 6)} people (4.0-6.0 people/m²) - Crowded but manageable")
    print(f"  • RED (Stampede): > {int(monitored_area * 6)} people (> 6.0 people/m²) - High risk of stampede")

    # Use a widely supported codec/container on Windows
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    # Start lightweight Flask server if requested
    global flask_app
    if args.serve:
        try:
            from flask import Flask, Response
        except Exception:
            raise RuntimeError("Flask not installed. Run: pip install flask")

        flask_app = Flask(__name__)

        @flask_app.route("/")
        def index():  # type: ignore
            return (
                """
                <html><head><title>Stampede Live</title></head>
                <body style='margin:0;background:#000;color:#fff;font-family:Arial'>
                <div style='padding:8px'>Live Stampede Risk Stream</div>
                <img src='/stream' style='width:100%;height:auto;display:block'/>
                </body></html>
                """
            )

        @flask_app.route("/stream")
        def stream():  # type: ignore
            def gen():
                while True:
                    if len(latest_jpeg) == 0:
                        # Avoid tight spin
                        cv2.waitKey(1)
                        continue
                    frame_bytes = latest_jpeg[0]
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

            return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

        def run_server():
            flask_app.run(host=args.host, port=args.port, debug=False, threaded=True)

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        print(f"[Stampede] Serving live preview at http://{args.host}:{args.port}")

    model = YOLO(weights)
    if args.device:
        try:
            model.to(args.device)
            print(f"[Stampede] Using device: {args.device}")
        except Exception as _:
            print(f"[Stampede] Could not set device '{args.device}', continuing on default")

    # Enhanced YOLOv11 tracking generator with optimized settings for best accuracy
    print("[Stampede] Starting enhanced YOLOv11 tracking stream...")
    print(f"[Stampede] Using confidence: {args.conf}, image size: {args.imgsz}, IoU: {args.iou_threshold}")
    print(f"[Stampede] Model: YOLOv11 Large - Best accuracy for dense crowd detection")
    results_stream = model.track(
        source=video_source,
        persist=True,
        stream=True,
        conf=args.conf,
        imgsz=args.imgsz,
        classes=[0],  # person class only
        verbose=False,
        task="detect",
        iou=args.iou_threshold,  # Lower IoU for better dense crowd detection
        max_det=args.max_detections,  # Allow more detections
        agnostic_nms=args.agnostic_nms,  # Better NMS for dense crowds
        augment=True,  # Test time augmentation for better accuracy
        half=True if args.device == 'cuda' else False,  # Use FP16 on GPU for speed
        save=False,
        save_txt=False,
        save_conf=False
    )

    # Density tracking for temporal smoothing
    density_history: List[np.ndarray] = []
    people_count_history: List[int] = []  # Track people count for stability
    frame_index = 0

    display_enabled = bool(args.display)
    
    # Setup display window with proper scaling
    if display_enabled:
        cv2.namedWindow("Stampede Risk", cv2.WINDOW_NORMAL)
        
        if args.fullscreen:
            # Set fullscreen mode with proper scaling
            cv2.setWindowProperty("Stampede Risk", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # Allow the window to resize and maintain aspect ratio
            cv2.setWindowProperty("Stampede Risk", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
        else:
            # For windowed mode, allow resizing
            cv2.resizeWindow("Stampede Risk", width, height)

    try:
        for result in results_stream:
            # Get original frame that YOLO used for this result
            if result.orig_img is None:
                continue
            frame_bgr = result.orig_img.copy()
            frame_index += 1

            # Enhanced person detection extraction with better handling
            centers: List[Tuple[int, int]] = []
            detection_boxes: List[Tuple[int, int, int, int]] = []  # Store full boxes for visualization
            confidence_scores: List[float] = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None
                conf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
                
                for i, box in enumerate(xyxy):
                    if cls is not None:
                        # 0 is 'person' in COCO
                        if int(cls[i]) != 0:
                            continue
                    
                    # Additional confidence filtering for dense crowds
                    if conf is not None and conf[i] < args.conf:
                        continue
                        
                    x0, y0, x1, y1 = box.astype(int)
                    
                    # Ensure boxes are within frame bounds (fix cropping issues)
                    h, w = frame_bgr.shape[:2]
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

            # Compute crowd density in people per square meter
            density_map = compute_density_map(centers, frame_bgr.shape, args.grid_w, args.grid_h, monitored_area)
            
            # Apply temporal smoothing to reduce false positives
            smoothed_density = smooth_density_temporal(density_map, density_history, args.smooth_frames)
            
    # Advanced crowd flow analysis
    flow_data = {'flow_intensity': 0.0, 'movement_direction': 'stable', 'crowd_velocity': 0.0}
    if args.crowd_flow:
        flow_data = analyze_crowd_flow(centers, frame_bgr.shape)
    
    # Advanced movement analysis
    movement_analysis = {'movement_risk_level': 'low', 'movement_risk_score': 0.0, 'movement_risk_factors': []}
    if args.movement_analysis:
        global movement_analyzer
        if movement_analyzer is None:
            movement_analyzer = MovementAnalyzer(history_size=30, flow_scale=args.flow_scale)
            # Adjust sensitivity based on user setting
            movement_analyzer.domino_threshold *= args.movement_sensitivity
            movement_analyzer.bottleneck_threshold *= args.movement_sensitivity
            movement_analyzer.panic_threshold *= (2.0 - args.movement_sensitivity)  # Invert for panic
            movement_analyzer.wave_threshold *= args.movement_sensitivity
        
        movement_analysis = movement_analyzer.analyze_movement_patterns(frame_bgr, centers, smoothed_density)
    
    # Advanced risk assessment
    risk_assessment = {'risk_score': 0.0, 'risk_level': 'low', 'risk_factors': []}
    if args.risk_assessment:
        risk_assessment = assess_risk_factors(num_people, overall_density, max_density, flow_data, args)

    # Calculate OVERALL density with stability (reduce fluctuation)
    num_people = len(centers)
    
    # Add to people count history for smoothing
    people_count_history.append(num_people)
    if len(people_count_history) > args.smooth_frames // 3:  # Keep shorter history for people count
        people_count_history.pop(0)
    
    # Use smoothed people count to reduce fluctuation
    smoothed_people_count = sum(people_count_history) / len(people_count_history)
    overall_density = smoothed_people_count / monitored_area if monitored_area > 0 else 0.0

    # Clean feed - no heatmap overlay, just original frame
    vis_bgr = frame_bgr.copy()

    # Enhanced visualization with bounding boxes and density indicators
    if len(detection_boxes) > 0:
        cell_w = max(1, frame_bgr.shape[1] // args.grid_w)
        cell_h = max(1, frame_bgr.shape[0] // args.grid_h)
        
        for i, (x0, y0, x1, y1) in enumerate(detection_boxes):
            cx = int((x0 + x1) * 0.5)
            cy = int((y0 + y1) * 0.5)
            gx = min(args.grid_w - 1, max(0, cx // cell_w))
            gy = min(args.grid_h - 1, max(0, cy // cell_h))
            local_density = float(smoothed_density[gy, gx]) if smoothed_density.size else 0.0
            
            # Enhanced color coding based on both overall and local density
            if overall_density >= args.danger_density or local_density >= args.danger_density:
                color = (0, 0, 255)  # red: danger zone
                thickness = 3
            elif overall_density >= args.warning_density or local_density >= args.warning_density:
                color = (0, 255, 255)  # yellow: warning zone
                thickness = 2
            else:
                color = (0, 200, 0)  # green: safe zone
                thickness = 1
            
            # Draw bounding box with confidence
            cv2.rectangle(vis_bgr, (x0, y0), (x1, y1), color, thickness)
            
            # Draw center dot (much smaller)
            cv2.circle(vis_bgr, (cx, cy), 3, color, -1)
            cv2.circle(vis_bgr, (cx, cy), 3, (255, 255, 255), 1)
            
            # Add confidence score for debugging (small text)
            if args.test_mode:
                conf_text = f"{confidence_scores[i]:.2f}"
                cv2.putText(vis_bgr, conf_text, (x0, y0-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Skip grid alerts to keep feed clean - only show person dots

    # HUD text with density-based status (overall_density already calculated above)
    max_density = float(np.max(smoothed_density)) if smoothed_density.size else 0.0
    avg_density = float(np.mean(smoothed_density)) if smoothed_density.size else 0.0
    
    # Enhanced status determination with advanced risk assessment
    
    # Use advanced risk assessment if enabled
    if args.risk_assessment and risk_assessment['risk_level'] != 'low':
        risk_level = risk_assessment['risk_level']
        risk_score = risk_assessment['risk_score']
        
        if risk_level == 'critical':
            status = f"🚨 CRITICAL RISK: {risk_score:.1f}"
            status_color = (0, 0, 255)  # red
        elif risk_level == 'high':
            status = f"⚠️ HIGH RISK: {risk_score:.1f}"
            status_color = (0, 0, 200)  # dark red
        elif risk_level == 'moderate':
            status = f"👥 MODERATE RISK: {risk_score:.1f}"
            status_color = (0, 165, 255)  # orange
        else:
            status = "✅ SAFE: NORMAL CONDITIONS"
            status_color = (0, 200, 0)  # green
    else:
        # Fallback to basic multi-factor assessment
        density_factor = max(overall_density, max_density)
        people_factor = num_people
        
        # User-specified thresholds: 6 people/m² for stampede, 4 for crowded
        # Use overall density for main status, max local density for alerts
        if overall_density >= args.danger_density:  # >= 6 people/m² overall
            status = "DANGER: STAMPEDE RISK"
            status_color = (0, 0, 255)  # red
        elif overall_density >= args.warning_density:  # >= 4 people/m² overall
            status = "CROWDED: MONITOR CLOSELY"
            status_color = (0, 255, 255)  # yellow
        else:
            status = "SAFE: NORMAL CONDITIONS"
            status_color = (0, 200, 0)  # green
    
    # Enhanced alert system with cooldown
    global last_alert_time
    current_time = time.time()
    
    # Check if we should trigger an alert
    should_alert = False
    alert_message = ""
    
    if args.risk_assessment and risk_assessment['risk_level'] in ['critical', 'high']:
        if current_time - last_alert_time > args.alert_cooldown:
            should_alert = True
            alert_message = f"ALERT: {risk_assessment['risk_level'].upper()} RISK DETECTED - Score: {risk_assessment['risk_score']:.2f}"
            if risk_assessment['risk_factors']:
                alert_message += f" - Factors: {', '.join(risk_assessment['risk_factors'])}"
            last_alert_time = current_time
    elif overall_density >= args.danger_density and num_people >= 5:
        if current_time - last_alert_time > args.alert_cooldown:
            should_alert = True
            alert_message = f"ALERT: HIGH DENSITY DETECTED - {overall_density:.2f} people/m² with {num_people} people"
            last_alert_time = current_time
    
    if should_alert:
        print(f"[ALERT] {alert_message}")
        # In a real system, this would trigger notifications, alarms, etc.
    
    # Debug info for accuracy
    if args.test_mode and frame_index % 30 == 0:
        debug_info = f"[DEBUG] Overall: {overall_density:.2f}/m², Max local: {max_density:.2f}/m², Status: {status}"
        if args.crowd_flow:
            debug_info += f", Flow: {flow_data['flow_intensity']:.2f}"
        if args.risk_assessment:
            debug_info += f", Risk: {risk_assessment['risk_score']:.2f} ({risk_assessment['risk_level']})"
        if args.movement_analysis:
            debug_info += f", Movement: {movement_analysis['movement_risk_level']} ({movement_analysis['movement_risk_score']:.2f})"
        print(debug_info)

    # Enhanced HUD with upper bar and range information
    hud_height = 200 if (args.test_mode or args.risk_assessment or args.movement_analysis) else 160
    cv2.rectangle(vis_bgr, (10, 10), (650, hud_height), (0, 0, 0), -1)
    cv2.rectangle(vis_bgr, (10, 10), (650, hud_height), (255, 255, 255), 1)
    
    # Upper bar with range information
    range_low = max(0, num_people)
    range_high_10 = num_people + 10
    range_high_20 = num_people + 20
    
    # Calculate average confidence for range estimation
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
    cv2.putText(vis_bgr, f"People Detected: {num_people}", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_bgr, f"{range_text}", (20, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, range_color, 2)
    cv2.putText(vis_bgr, f"Avg Confidence: {avg_confidence:.2f}", (20, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(vis_bgr, f"Density: {overall_density:.2f} people/m²", (20, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_bgr, f"Status: {status}", (20, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
    
    # Advanced features info
    y_offset = 130
    if args.crowd_flow:
        cv2.putText(vis_bgr, f"Flow: {flow_data['movement_direction']} ({flow_data['flow_intensity']:.2f})", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
    
    if args.risk_assessment:
        cv2.putText(vis_bgr, f"Risk Score: {risk_assessment['risk_score']:.2f} ({risk_assessment['risk_level']})", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
    
    # Movement analysis info
    if args.movement_analysis:
        cv2.putText(vis_bgr, f"Movement Risk: {movement_analysis['movement_risk_level']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
    
    # Additional info for test mode
    if args.test_mode:
        cv2.putText(vis_bgr, f"Max Local: {max_density:.2f}/m²", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(vis_bgr, f"Avg Local: {avg_density:.2f}/m²", (20, y_offset + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Range warning indicator
    if avg_confidence < 0.6 and num_people > 0:
        warning_text = "⚠️ Possible missed detections - Check range!"
        cv2.putText(vis_bgr, warning_text, (20, y_offset + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    # Progress bar line
    if total_frames > 0:
        progress = frame_index / max(total_frames, 1)
        bar_w = int(0.9 * width)
        x0 = int(0.05 * width)
        y0 = height - 20
        cv2.rectangle(vis_bgr, (x0, y0), (x0 + bar_w, y0 + 10), (60, 60, 60), -1)
        cv2.rectangle(vis_bgr, (x0, y0), (x0 + int(bar_w * progress), y0 + 10), (0, 200, 255), -1)

    # Optional frame skipping for heavy pipelines
    if args.skip_frames > 0 and (frame_index % (args.skip_frames + 1)) != 1:
        pass
    else:
        out.write(vis_bgr)

    if frame_index % 30 == 0:
        if args.test_mode:
            print(f"[TEST] {num_people} people in {monitored_area:.1f}m² = {overall_density:.2f}/m² → {status}")
        else:
            print(f"[Stampede] {num_people} people detected, overall density: {overall_density:.2f}/m²")

    # Publish to live server if enabled
    if args.serve:
        # Throttle stream updates to reduce bandwidth and CPU
        if not hasattr(main, "_last_stream_time"):
            main._last_stream_time = 0.0  # type: ignore[attr-defined]
        now = time.time()
        min_interval = 1.0 / max(1, args.max_stream_fps)
        if now - main._last_stream_time >= min_interval:
            ret, jpeg = cv2.imencode('.jpg', vis_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), max(10, min(95, args.jpeg_quality))])
            if ret:
                latest_jpeg.clear()
                latest_jpeg.append(jpeg.tobytes())
                main._last_stream_time = now  # type: ignore[attr-defined]

    if display_enabled:
        try:
            # For fullscreen mode, ensure the frame fills the window properly
            if args.fullscreen:
                # Get current window size to scale frame if needed
                window_rect = cv2.getWindowImageRect("Stampede Risk")
                if window_rect[2] > 0 and window_rect[3] > 0:  # Valid window dimensions
                    window_width, window_height = window_rect[2], window_rect[3]
                    # Only resize if window dimensions differ significantly from frame
                    if abs(window_width - vis_bgr.shape[1]) > 50 or abs(window_height - vis_bgr.shape[0]) > 50:
                        display_frame = cv2.resize(vis_bgr, (window_width, window_height))
                    else:
                        display_frame = vis_bgr
                else:
                    display_frame = vis_bgr
            else:
                display_frame = vis_bgr
            
            cv2.imshow("Stampede Risk", display_frame)
            # Keep UI latency minimal; don't oversleep
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except cv2.error:
            # GUI backend not available (e.g., headless OpenCV). Disable live display and continue saving output.
            print("[Stampede] OpenCV GUI not available. Disabling live display.")
            display_enabled = False
    except Exception as e:
        print(f"[Stampede] Error during processing: {e}")

    out.release()
    cap.release()
    if display_enabled:
        cv2.destroyAllWindows()
    if frame_index == 0:
        print("[Stampede] No frames were processed. Possible causes: unsupported codec, corrupted file, or path issue.")
        print("[Stampede] Try re-encoding the input, e.g.: ffmpeg -y -i input.mp4 -c:v libx264 -crf 20 -pix_fmt yuv420p demo_video2.mp4")

    print(f"[Stampede] Saved: {args.out}")


if __name__ == "__main__":
    main()


