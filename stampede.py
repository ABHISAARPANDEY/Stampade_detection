import argparse
import os
from typing import List, Tuple, Dict
import threading
from collections import deque
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Shared buffer for live streaming
latest_jpeg = deque(maxlen=1)
flask_app = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stampede risk detection with YOLOv8 tracking")
    parser.add_argument("--video", type=str, default="demo_video2.mp4", help="Path to input video")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to YOLO weights (.pt). If not provided, tries ./training/... then falls back to bundled yolov8n.pt",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size")
    parser.add_argument("--out", type=str, default="stampede_output.mp4", help="Path to save annotated video")
    parser.add_argument("--grid_w", type=int, default=20, help="Number of grid cells horizontally")
    parser.add_argument("--grid_h", type=int, default=12, help="Number of grid cells vertically")
    parser.add_argument("--risk_thresh", type=float, default=0.6, help="Threshold for risk alert [0,1]")
    parser.add_argument("--medium_thresh", type=float, default=0.3, help="Medium risk threshold for yellow boxes [0,1]")
    parser.add_argument("--display", action="store_true", help="Show live preview window")
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
    return parser.parse_args()


def select_weights(user_weights: str | None) -> str:
    if user_weights and os.path.exists(user_weights):
        return user_weights
    # Try project training path first
    candidates: List[str] = [
        "./training/yolov8m/train/weights/best.pt",
        "./training/yolov8s/train/weights/best.pt",
        "./training/yolov8n/train/weights/best.pt",
        "./yolov8m.pt",
        "./yolov8s.pt",
        "./yolov8n.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Last resort: rely on ultralytics' default pretrained (will download if needed)
    return "yolov8n.pt"


def compute_density_map(
    centers: List[Tuple[int, int]], frame_shape: Tuple[int, int, int], grid_w: int, grid_h: int
) -> np.ndarray:
    h, w = frame_shape[:2]
    density = np.zeros((grid_h, grid_w), dtype=np.float32)
    if not centers:
        return density
    cell_w = max(1, w // grid_w)
    cell_h = max(1, h // grid_h)
    for cx, cy in centers:
        gx = min(grid_w - 1, max(0, cx // cell_w))
        gy = min(grid_h - 1, max(0, cy // cell_h))
        density[gy, gx] += 1.0
    # Smooth spatially to better reflect local crowding
    density = cv2.GaussianBlur(density, (3, 3), 0)
    return density


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


def normalize_map(x: np.ndarray, ema_state: dict, key: str, momentum: float = 0.9) -> np.ndarray:
    current_max = float(np.max(x)) if x.size else 0.0
    ema_key = f"max_{key}"
    if ema_key not in ema_state:
        ema_state[ema_key] = current_max if current_max > 0 else 1.0
    else:
        ema_state[ema_key] = momentum * ema_state[ema_key] + (1 - momentum) * max(current_max, 1e-6)
    denom = max(ema_state[ema_key], 1e-6)
    return np.clip(x / denom, 0.0, 1.0)


def overlay_heatmap(base_bgr: np.ndarray, risk_map: np.ndarray) -> np.ndarray:
    h, w = base_bgr.shape[:2]
    risk_resized = cv2.resize(risk_map, (w, h), interpolation=cv2.INTER_LINEAR)
    risk_uint8 = (np.clip(risk_resized, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(risk_uint8, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(base_bgr, 0.7, heat, 0.3, 0)
    return overlay


def draw_grid_alerts(
    frame: np.ndarray, risk_map: np.ndarray, thresh: float, color: Tuple[int, int, int] = (0, 0, 255)
) -> None:
    h, w = frame.shape[:2]
    gh, gw = risk_map.shape[:2]
    cell_w = w // gw
    cell_h = h // gh
    for gy in range(gh):
        for gx in range(gw):
            risk = float(risk_map[gy, gx])
            if risk >= thresh:
                x0 = gx * cell_w
                y0 = gy * cell_h
                x1 = w if gx == gw - 1 else (gx + 1) * cell_w
                y1 = h if gy == gh - 1 else (gy + 1) * cell_h
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)


def main() -> None:
    args = parse_args()
    weights = select_weights(args.weights)

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Input video not found: {args.video}")

    print(f"[Stampede] Using weights: {weights}")
    print(f"[Stampede] Opening video: {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Stampede] Video info: {width}x{height} @ {fps:.2f} FPS, frames={total_frames}")

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

    # Tracking generator
    print("[Stampede] Starting YOLO tracking stream...")
    results_stream = model.track(
        source=args.video,
        persist=True,
        stream=True,
        conf=args.conf,
        imgsz=args.imgsz,
        classes=[0],  # person class only
        verbose=False,
        task="detect",
    )

    ema_state: dict = {}
    prev_gray: np.ndarray | None = None
    # For motion by tracks
    prev_centers_by_id: Dict[int, Tuple[int, int]] = {}
    frame_index = 0

    display_enabled = bool(args.display)

    try:
        for result in results_stream:
            # Get original frame that YOLO used for this result
            if result.orig_img is None:
                continue
            frame_bgr = result.orig_img.copy()
            frame_index += 1

            # Extract person detections
            centers: List[Tuple[int, int]] = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None
                for i, box in enumerate(xyxy):
                    if cls is not None:
                        # 0 is 'person' in COCO
                        if int(cls[i]) != 0:
                            continue
                    x0, y0, x1, y1 = box.astype(int)
                    cx = int((x0 + x1) * 0.5)
                    cy = int((y0 + y1) * 0.5)
                    centers.append((cx, cy))

            # Density map (people per grid cell)
            density_map = compute_density_map(centers, frame_bgr.shape, args.grid_w, args.grid_h)
            density_norm = normalize_map(density_map, ema_state, key="density", momentum=0.95)

            # Motion signal: tracks (fast) or optical flow (heavier)
            motion_map = np.zeros((args.grid_h, args.grid_w), dtype=np.float32)
            if args.motion == "tracks":
                # Build current centers by track id if available
                centers_by_id: Dict[int, Tuple[int, int]] = {}
                if result.boxes is not None and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    ids = result.boxes.id.int().cpu().numpy()
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    cls = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None
                    for i, box in enumerate(xyxy):
                        if cls is not None and int(cls[i]) != 0:
                            continue
                        tid = int(ids[i])
                        x0, y0, x1, y1 = box.astype(int)
                        cx = int((x0 + x1) * 0.5)
                        cy = int((y0 + y1) * 0.5)
                        centers_by_id[tid] = (cx, cy)

                # Convert motion vectors to grid magnitudes
                h, w = frame_bgr.shape[:2]
                cell_w = max(1, w // args.grid_w)
                cell_h = max(1, h // args.grid_h)
                for tid, (cx, cy) in centers_by_id.items():
                    if tid in prev_centers_by_id:
                        px, py = prev_centers_by_id[tid]
                        vx = cx - px
                        vy = cy - py
                        speed = float(np.hypot(vx, vy))
                        gx = min(args.grid_w - 1, max(0, cx // cell_w))
                        gy = min(args.grid_h - 1, max(0, cy // cell_h))
                        motion_map[gy, gx] += speed
                motion_map = cv2.GaussianBlur(motion_map, (3, 3), 0)
                prev_centers_by_id = centers_by_id
            else:
                # Optical flow path (kept for completeness)
                gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                flow = None
                do_flow = (frame_index % max(1, args.flow_every)) == 0
                if do_flow:
                    if 0.25 <= args.flow_scale < 1.0:
                        small_size = (max(2, int(frame_bgr.shape[1] * args.flow_scale)),
                                      max(2, int(frame_bgr.shape[0] * args.flow_scale)))
                        gray = cv2.resize(gray_full, small_size, interpolation=cv2.INTER_AREA)
                        prev_src = prev_gray if prev_gray is None else cv2.resize(prev_gray, small_size, interpolation=cv2.INTER_AREA)
                    else:
                        gray = gray_full
                        prev_src = prev_gray

                    if prev_src is not None:
                        flow_small = cv2.calcOpticalFlowFarneback(
                            prev_src,
                            gray,
                            None,
                            pyr_scale=0.5,
                            levels=3,
                            winsize=15,
                            iterations=2,
                            poly_n=5,
                            poly_sigma=1.1,
                            flags=0,
                        )
                        flow = cv2.resize(flow_small, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                    prev_gray = gray_full
                motion_map = compute_motion_map(flow, args.grid_w, args.grid_h)
            motion_norm = normalize_map(motion_map, ema_state, key="motion", momentum=0.95)

            # Combined risk: crowding x motion
            risk_map = np.clip(density_norm * motion_norm, 0.0, 1.0)

            # Start from original frame, overlay heatmap, then draw risk-colored boxes
            vis_bgr = overlay_heatmap(frame_bgr.copy(), risk_map)

            # Draw per-person boxes colored by local risk
            if result.boxes is not None and len(result.boxes) > 0:
                cell_w = max(1, frame_bgr.shape[1] // args.grid_w)
                cell_h = max(1, frame_bgr.shape[0] // args.grid_h)
                for i, box in enumerate(xyxy):
                    if cls is not None and int(cls[i]) != 0:
                        continue
                    x0, y0, x1, y1 = box.astype(int)
                    cx = int((x0 + x1) * 0.5)
                    cy = int((y0 + y1) * 0.5)
                    gx = min(args.grid_w - 1, max(0, cx // cell_w))
                    gy = min(args.grid_h - 1, max(0, cy // cell_h))
                    person_risk = float(risk_map[gy, gx]) if risk_map.size else 0.0
                    if person_risk >= args.risk_thresh:
                        color = (0, 0, 255)  # red: high risk
                    elif person_risk >= args.medium_thresh:
                        color = (0, 255, 255)  # yellow: medium risk
                    else:
                        color = (0, 200, 0)  # green: low risk
                    cv2.rectangle(vis_bgr, (x0, y0), (x1, y1), color, 3)
                    cv2.putText(
                        vis_bgr,
                        f"r:{person_risk:.2f}",
                        (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            # Draw grid alerts after boxes
            draw_grid_alerts(vis_bgr, risk_map, args.risk_thresh)

            # HUD text
            num_people = len(centers)
            overall_risk = float(np.mean(risk_map)) if risk_map.size else 0.0
            status = "ALERT: HIGH RISK" if overall_risk >= args.risk_thresh else "RISK: NORMAL"
            status_color = (0, 0, 255) if overall_risk >= args.risk_thresh else (0, 200, 0)

            cv2.rectangle(vis_bgr, (10, 10), (360, 90), (0, 0, 0), -1)
            cv2.putText(vis_bgr, f"People: {num_people}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(
                vis_bgr,
                f"Risk: {overall_risk:.2f}  {status}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
            )

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
                avg_risk = float(np.mean(risk_map)) if risk_map.size else 0.0
                print(f"[Stampede] Processed {frame_index} frames, avg risk={avg_risk:.2f}")

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
                    cv2.imshow("Stampede Risk", vis_bgr)
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


