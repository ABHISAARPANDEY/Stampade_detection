## Stampede Risk Detector – Simple Guide

This document explains, in plain language, how the stampede detector works, what you’ll see on screen, and how to run and tune it. No ML background required.

### What it does
- Watches a video (or live camera) and finds people.
- Estimates where the crowd is getting dense and moving fast in the same area.
- Highlights risky areas on the video and colors each person’s box:
  - Red = high risk
  - Yellow = medium risk
  - Green = low risk
- Shows a heatmap over the frame and outlines risky grid cells.
- Displays a small “HUD” with total people and overall risk.

### Why this works (intuition)
Stampedes usually happen where two things combine:
1) Too many people packed together (crowd density)
2) People moving quickly or being pushed (motion)

The detector measures both at once. If a place in the video has lots of people and those people are moving more than normal, risk goes up.

### How it works (high level)
1) Person detection (YOLOv8): Finds people in each frame and draws boxes.
2) Tracking: Gives the same person a consistent ID across frames.
3) Density map: Splits the frame into a grid (like a chessboard) and counts how many people are in each cell.
4) Motion map: Uses person tracking to estimate how fast people move in each cell (how much the boxes shift between frames).
5) Risk map: Multiplies density × motion. High density and high motion together = higher risk.
6) Visuals:
   - Heatmap shows risk across the whole frame.
   - Grid cells with high risk are outlined.
   - Each person’s box is colored by the local risk (red/yellow/green) and labeled with a small risk number.

### What you’ll see on screen
- Bounding boxes around people: their color reflects local risk.
- Heatmap glow: brighter (hotter) areas = more risk.
- Grid rectangles: red outline on cells above the high-risk threshold.
- HUD: number of people, overall risk, and a status (NORMAL or ALERT: HIGH RISK).

### How to run
Basic (no GPU required):
```
pip install -r requirements.txt
python stampede.py --video demo_video2.mp4 --display --out stampede_output.mp4
```
- Press q to close the live window.
- The annotated video saves to `stampede_output.mp4`.

Faster with NVIDIA GPU (optional):
```
python stampede.py --video demo_video2.mp4 --display --out stampede_output.mp4 --device 0 --imgsz 640 --motion tracks
```

### Tuning (simple knobs)
- Risk sensitivity:
  - `--medium_thresh 0.3` (yellow starts here)
  - `--risk_thresh 0.6` (red starts here)
  - Lower these to see more yellow/red; raise to be stricter.
- Smoothness vs. load (if preview lags):
  - Keep `--motion tracks` (already fast). If still needed, try `--imgsz 512`.
- Appearance:
  - Grid size: `--grid_w 20 --grid_h 12` (more cells = finer heatmap but a bit more compute)

### What “risk” means here
- Risk is a unitless score between 0 and 1 for each grid cell.
- It’s the combination of how crowded and how much motion is happening.
- It’s not a guarantee of a stampede; it’s a warning indicator of conditions that often precede one.

### Typical use cases
- Live monitoring of entrances, corridors, stairs, or narrow passages.
- Stadiums, stations, malls, events—places where crowding/flow changes quickly.
- Drones: run headless on the on-board computer, send alerts and an annotated stream to the operator.

### Good practices & limitations
- Camera placement matters: a wide, stable view helps the model judge density and motion.
- Lighting and weather affect detection; very dark or blurry scenes reduce accuracy.
- Extremely crowded scenes with heavy occlusion are hard; alerts help focus attention but are not guarantees.
- Use alerts as decision support, not as the only trigger for action.

### Troubleshooting quick tips
- No window opens: ensure non-headless OpenCV is installed (you did this if the window appears).
- Laggy preview:
  - Use `--device 0` (GPU), keep `--motion tracks`.
  - Optionally lower `--imgsz` slightly (e.g., 512) if needed.
- No output saved: check you have write permission and enough disk space.

### Short glossary
- YOLOv8: A fast, modern object detector that finds people in each frame.
- Tracking ID: A number that follows the same person across frames.
- Density map: A grid counting how many people are in each cell.
- Motion (tracks): How much the same person’s box moves between frames.
- Risk map: Density × motion; used to color boxes and highlight areas.

### One-line summary
“We color each person and area by how crowded and how fast people are moving there. Where both are high, we warn you early.”


