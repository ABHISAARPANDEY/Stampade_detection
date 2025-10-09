"""
Heat Map Visualizer for STAMPede Detection System
Creates real-time density heat maps with advanced visualization options
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import time

class HeatMapStyle(Enum):
    INFERNO = "inferno"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    JET = "jet"
    HOT = "hot"
    COOL = "cool"
    CUSTOM = "custom"

@dataclass
class HeatMapConfig:
    style: HeatMapStyle = HeatMapStyle.INFERNO
    alpha: float = 0.6
    blur_radius: int = 5
    min_density: float = 0.0
    max_density: float = 10.0
    show_contours: bool = True
    contour_levels: int = 10
    show_colorbar: bool = True
    colorbar_position: str = "right"
    grid_overlay: bool = False
    grid_alpha: float = 0.3
    show_peaks: bool = True
    peak_threshold: float = 0.8

class HeatMapVisualizer:
    """Advanced heat map visualization for crowd density analysis"""
    
    def __init__(self, config: Optional[HeatMapConfig] = None):
        self.config = config or HeatMapConfig()
        self.custom_colormap = None
        self._setup_custom_colormap()
    
    def _setup_custom_colormap(self):
        """Setup custom colormap for stampede detection"""
        # Custom colormap: Green (safe) -> Yellow (warning) -> Red (danger)
        colors = ['#00ff00', '#ffff00', '#ff8000', '#ff0000', '#800000']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('stampede', colors, N=n_bins)
        self.custom_colormap = cmap
    
    def create_density_heatmap(self, density_map: np.ndarray, 
                             frame_shape: Tuple[int, int, int],
                             config: Optional[HeatMapConfig] = None) -> np.ndarray:
        """Create a density heat map overlay"""
        if config is None:
            config = self.config
        
        h, w = frame_shape[:2]
        gh, gw = density_map.shape
        
        # Resize density map to frame size
        density_resized = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize density to 0-1 range
        density_normalized = np.clip(
            (density_resized - config.min_density) / 
            (config.max_density - config.min_density), 0, 1
        )
        
        # Apply Gaussian blur for smoother visualization
        if config.blur_radius > 0:
            density_normalized = cv2.GaussianBlur(
                density_normalized, 
                (config.blur_radius * 2 + 1, config.blur_radius * 2 + 1), 
                0
            )
        
        # Convert to heat map
        heat_map = self._apply_colormap(density_normalized, config)
        
        # Create alpha mask
        alpha_mask = np.zeros((h, w), dtype=np.float32)
        alpha_mask[density_normalized > 0] = config.alpha
        
        # Apply alpha blending
        heat_map = heat_map.astype(np.float32) / 255.0
        alpha_mask = alpha_mask[..., np.newaxis]
        
        return heat_map, alpha_mask
    
    def _apply_colormap(self, normalized_density: np.ndarray, 
                       config: HeatMapConfig) -> np.ndarray:
        """Apply colormap to normalized density"""
        if config.style == HeatMapStyle.CUSTOM:
            colormap = self.custom_colormap
        else:
            colormap = getattr(cm, config.style.value)
        
        # Convert to RGB
        heat_map = colormap(normalized_density)[:, :, :3]
        heat_map = (heat_map * 255).astype(np.uint8)
        
        return heat_map
    
    def overlay_heatmap(self, frame: np.ndarray, density_map: np.ndarray,
                       config: Optional[HeatMapConfig] = None) -> np.ndarray:
        """Overlay heat map on frame"""
        if config is None:
            config = self.config
        
        heat_map, alpha_mask = self.create_density_heatmap(density_map, frame.shape, config)
        
        # Convert frame to float
        frame_float = frame.astype(np.float32) / 255.0
        
        # Blend heat map with frame
        result = frame_float * (1 - alpha_mask) + heat_map * alpha_mask
        
        # Convert back to uint8
        result = (result * 255).astype(np.uint8)
        
        # Add contours if enabled
        if config.show_contours:
            result = self._add_contours(result, density_map, config)
        
        # Add grid overlay if enabled
        if config.grid_overlay:
            result = self._add_grid_overlay(result, density_map, config)
        
        # Add peak markers if enabled
        if config.show_peaks:
            result = self._add_peak_markers(result, density_map, config)
        
        return result
    
    def _add_contours(self, frame: np.ndarray, density_map: np.ndarray,
                     config: HeatMapConfig) -> np.ndarray:
        """Add contour lines to the heat map"""
        h, w = frame.shape[:2]
        gh, gw = density_map.shape
        
        # Resize density map to frame size
        density_resized = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize for contour levels
        density_normalized = np.clip(
            (density_resized - config.min_density) / 
            (config.max_density - config.min_density), 0, 1
        )
        
        # Create contour levels
        levels = np.linspace(0, 1, config.contour_levels + 1)
        
        # Find contours
        contours, _ = cv2.findContours(
            (density_normalized * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Draw contours
        contour_frame = frame.copy()
        cv2.drawContours(contour_frame, contours, -1, (255, 255, 255), 1)
        
        return contour_frame
    
    def _add_grid_overlay(self, frame: np.ndarray, density_map: np.ndarray,
                         config: HeatMapConfig) -> np.ndarray:
        """Add grid overlay to show density cells"""
        h, w = frame.shape[:2]
        gh, gw = density_map.shape
        
        grid_frame = frame.copy()
        
        # Calculate grid cell size
        cell_w = w // gw
        cell_h = h // gh
        
        # Draw vertical lines
        for i in range(1, gw):
            x = i * cell_w
            cv2.line(grid_frame, (x, 0), (x, h), (255, 255, 255), 1)
        
        # Draw horizontal lines
        for i in range(1, gh):
            y = i * cell_h
            cv2.line(grid_frame, (0, y), (w, y), (255, 255, 255), 1)
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 1 - config.grid_alpha, grid_frame, config.grid_alpha, 0)
        
        return result
    
    def _add_peak_markers(self, frame: np.ndarray, density_map: np.ndarray,
                         config: HeatMapConfig) -> np.ndarray:
        """Add markers for density peaks"""
        h, w = frame.shape[:2]
        gh, gw = density_map.shape
        
        # Find peaks
        peaks = self._find_density_peaks(density_map, config.peak_threshold)
        
        # Draw peak markers
        result = frame.copy()
        for peak_y, peak_x in peaks:
            # Convert grid coordinates to frame coordinates
            frame_x = int((peak_x + 0.5) * w / gw)
            frame_y = int((peak_y + 0.5) * h / gh)
            
            # Draw marker
            cv2.circle(result, (frame_x, frame_y), 8, (0, 0, 255), 2)
            cv2.circle(result, (frame_x, frame_y), 4, (255, 255, 255), -1)
        
        return result
    
    def _find_density_peaks(self, density_map: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
        """Find local maxima in density map"""
        from scipy.ndimage import maximum_filter
        
        # Find local maxima
        local_maxima = maximum_filter(density_map, size=3) == density_map
        
        # Apply threshold
        peaks = np.where((local_maxima) & (density_map > threshold * np.max(density_map)))
        
        return list(zip(peaks[0], peaks[1]))
    
    def create_standalone_heatmap(self, density_map: np.ndarray,
                                config: Optional[HeatMapConfig] = None) -> np.ndarray:
        """Create a standalone heat map without frame overlay"""
        if config is None:
            config = self.config
        
        # Normalize density
        density_normalized = np.clip(
            (density_map - config.min_density) / 
            (config.max_density - config.min_density), 0, 1
        )
        
        # Apply colormap
        heat_map = self._apply_colormap(density_normalized, config)
        
        # Add contours if enabled
        if config.show_contours:
            heat_map = self._add_contours(heat_map, density_map, config)
        
        return heat_map
    
    def create_3d_heatmap(self, density_map: np.ndarray,
                         config: Optional[HeatMapConfig] = None) -> np.ndarray:
        """Create a 3D visualization of the density map"""
        if config is None:
            config = self.config
        
        # Create 3D surface
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate grids
        gh, gw = density_map.shape
        x = np.arange(gw)
        y = np.arange(gh)
        X, Y = np.meshgrid(x, y)
        
        # Plot surface
        if config.style == HeatMapStyle.CUSTOM:
            colormap = self.custom_colormap
        else:
            colormap = getattr(cm, config.style.value)
        
        surf = ax.plot_surface(X, Y, density_map, cmap=colormap, alpha=0.8)
        
        # Add colorbar
        if config.show_colorbar:
            fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # Set labels
        ax.set_xlabel('Grid X')
        ax.set_ylabel('Grid Y')
        ax.set_zlabel('Density (people/m²)')
        ax.set_title('3D Crowd Density Heat Map')
        
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return img
    
    def create_animated_heatmap(self, density_maps: List[np.ndarray],
                              config: Optional[HeatMapConfig] = None) -> List[np.ndarray]:
        """Create animated heat map from sequence of density maps"""
        if config is None:
            config = self.config
        
        frames = []
        for density_map in density_maps:
            frame = self.create_standalone_heatmap(density_map, config)
            frames.append(frame)
        
        return frames
    
    def get_density_statistics(self, density_map: np.ndarray) -> Dict[str, float]:
        """Get statistics about the density map"""
        stats = {
            'min_density': float(np.min(density_map)),
            'max_density': float(np.max(density_map)),
            'mean_density': float(np.mean(density_map)),
            'median_density': float(np.median(density_map)),
            'std_density': float(np.std(density_map)),
            'total_people': float(np.sum(density_map)),
            'dense_cells': int(np.sum(density_map > 4.0)),  # Cells with >4 people/m²
            'danger_cells': int(np.sum(density_map > 6.0))  # Cells with >6 people/m²
        }
        
        return stats
    
    def create_heatmap_legend(self, config: Optional[HeatMapConfig] = None) -> np.ndarray:
        """Create a color legend for the heat map"""
        if config is None:
            config = self.config
        
        # Create legend image
        legend_height = 200
        legend_width = 50
        
        # Create gradient
        gradient = np.linspace(0, 1, legend_height).reshape(-1, 1)
        gradient = np.repeat(gradient, legend_width, axis=1)
        
        # Apply colormap
        if config.style == HeatMapStyle.CUSTOM:
            colormap = self.custom_colormap
        else:
            colormap = getattr(cm, config.style.value)
        
        legend = colormap(gradient)[:, :, :3]
        legend = (legend * 255).astype(np.uint8)
        
        # Add labels
        legend_with_labels = np.zeros((legend_height + 40, legend_width + 100, 3), dtype=np.uint8)
        legend_with_labels[:legend_height, :legend_width] = legend
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (255, 255, 255)
        
        # Min label
        cv2.putText(legend_with_labels, f"{config.min_density:.1f}", 
                   (legend_width + 5, legend_height - 10), font, font_scale, color, 1)
        
        # Max label
        cv2.putText(legend_with_labels, f"{config.max_density:.1f}", 
                   (legend_width + 5, 10), font, font_scale, color, 1)
        
        # Units label
        cv2.putText(legend_with_labels, "people/m²", 
                   (legend_width + 5, legend_height + 20), font, font_scale, color, 1)
        
        return legend_with_labels
    
    def update_config(self, **kwargs):
        """Update heat map configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Recreate custom colormap if style changed
        if 'style' in kwargs and kwargs['style'] == HeatMapStyle.CUSTOM:
            self._setup_custom_colormap()
    
    def get_available_styles(self) -> List[str]:
        """Get list of available colormap styles"""
        return [style.value for style in HeatMapStyle]
