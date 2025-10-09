"""
Alert Manager for STAMPede Detection System
Handles real-time alerts, sound notifications, and push notifications
"""

import time
import threading
import json
import os
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import pygame
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"

class AlertType(Enum):
    DENSITY_ALERT = "density_alert"
    CROWD_FLOW_ALERT = "crowd_flow_alert"
    MOVEMENT_ALERT = "movement_alert"
    SYSTEM_ALERT = "system_alert"
    CUSTOM_ALERT = "custom_alert"

@dataclass
class AlertConfig:
    alert_type: AlertType
    alert_level: AlertLevel
    threshold_value: float
    cooldown_seconds: int = 30
    sound_enabled: bool = True
    email_enabled: bool = False
    sms_enabled: bool = False
    webhook_enabled: bool = False
    message_template: str = ""

@dataclass
class Alert:
    id: str
    timestamp: float
    camera_id: int
    alert_type: AlertType
    alert_level: AlertLevel
    message: str
    data: Dict[str, Any]
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None

class SoundManager:
    """Manages sound notifications and alerts"""
    
    def __init__(self):
        self.sounds = {}
        self.volume = 0.7
        self.enabled = True
        self._init_sounds()
    
    def _init_sounds(self):
        """Initialize sound effects"""
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # Create simple beep sounds programmatically
            self.sounds = {
                'info': self._create_beep_sound(440, 0.2),  # A4 note
                'warning': self._create_beep_sound(554, 0.3),  # C#5 note
                'danger': self._create_beep_sound(659, 0.5),  # E5 note
                'critical': self._create_urgent_sound()  # Urgent pattern
            }
            
            print("[SoundManager] Sound system initialized")
        except Exception as e:
            print(f"[SoundManager] Failed to initialize sound: {e}")
            self.enabled = False
    
    def _create_beep_sound(self, frequency: int, duration: float):
        """Create a simple beep sound"""
        try:
            sample_rate = 22050
            frames = int(duration * sample_rate)
            arr = []
            
            for i in range(frames):
                time_val = i / sample_rate
                wave = 4096 * (1 if (int(time_val * frequency) % 2) else -1)
                arr.append([int(wave), int(wave)])
            
            sound = pygame.sndarray.make_sound(arr)
            return sound
        except Exception as e:
            print(f"[SoundManager] Failed to create beep sound: {e}")
            return None
    
    def _create_urgent_sound(self):
        """Create an urgent alert sound pattern"""
        try:
            # Create a pattern of quick beeps
            sample_rate = 22050
            duration = 0.1
            frames = int(duration * sample_rate)
            arr = []
            
            for i in range(frames * 3):  # 3 beeps
                time_val = (i % frames) / sample_rate
                wave = 4096 * (1 if (int(time_val * 800) % 2) else -1)
                arr.append([int(wave), int(wave)])
            
            sound = pygame.sndarray.make_sound(arr)
            return sound
        except Exception as e:
            print(f"[SoundManager] Failed to create urgent sound: {e}")
            return None
    
    def play_alert(self, alert_level: AlertLevel):
        """Play alert sound based on level"""
        if not self.enabled or not self.sounds:
            return
        
        try:
            sound = self.sounds.get(alert_level.value)
            if sound:
                sound.set_volume(self.volume)
                sound.play()
        except Exception as e:
            print(f"[SoundManager] Failed to play sound: {e}")
    
    def set_volume(self, volume: float):
        """Set sound volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
    
    def set_enabled(self, enabled: bool):
        """Enable or disable sound alerts"""
        self.enabled = enabled

class EmailManager:
    """Manages email notifications"""
    
    def __init__(self, smtp_server: str = "", smtp_port: int = 587, 
                 username: str = "", password: str = ""):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.enabled = bool(smtp_server and username and password)
    
    def send_alert(self, alert: Alert, recipients: List[str]):
        """Send email alert"""
        if not self.enabled or not recipients:
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"STAMPede Alert - {alert.alert_level.value.upper()}"
            
            body = f"""
            STAMPede Detection System Alert
            
            Alert Level: {alert.alert_level.value.upper()}
            Camera ID: {alert.camera_id}
            Time: {datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}
            
            Message: {alert.message}
            
            Data:
            {json.dumps(alert.data, indent=2)}
            
            Please check the system immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(self.username, recipients, text)
            server.quit()
            
            print(f"[EmailManager] Alert email sent to {len(recipients)} recipients")
            return True
        except Exception as e:
            print(f"[EmailManager] Failed to send email: {e}")
            return False

class WebhookManager:
    """Manages webhook notifications"""
    
    def __init__(self):
        self.webhooks = []
        self.enabled = True
    
    def add_webhook(self, url: str, headers: Optional[Dict[str, str]] = None):
        """Add a webhook URL"""
        webhook = {
            'url': url,
            'headers': headers or {}
        }
        self.webhooks.append(webhook)
        print(f"[WebhookManager] Added webhook: {url}")
    
    def send_alert(self, alert: Alert):
        """Send webhook alert"""
        if not self.enabled or not self.webhooks:
            return
        
        payload = {
            'id': alert.id,
            'timestamp': alert.timestamp,
            'camera_id': alert.camera_id,
            'alert_type': alert.alert_type.value,
            'alert_level': alert.alert_level.value,
            'message': alert.message,
            'data': alert.data
        }
        
        for webhook in self.webhooks:
            try:
                response = requests.post(
                    webhook['url'],
                    json=payload,
                    headers=webhook['headers'],
                    timeout=10
                )
                if response.status_code == 200:
                    print(f"[WebhookManager] Webhook sent successfully to {webhook['url']}")
                else:
                    print(f"[WebhookManager] Webhook failed with status {response.status_code}")
            except Exception as e:
                print(f"[WebhookManager] Webhook error: {e}")

class AlertManager:
    """Main alert management system"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_configs: Dict[AlertType, AlertConfig] = {}
        self.last_alert_times: Dict[str, float] = {}
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Initialize managers
        self.sound_manager = SoundManager()
        self.email_manager = EmailManager()
        self.webhook_manager = WebhookManager()
        
        # Default alert configurations
        self._setup_default_configs()
        
        # Alert history
        self.alert_history: List[Alert] = []
        self.max_history = 1000
    
    def _setup_default_configs(self):
        """Setup default alert configurations"""
        configs = [
            AlertConfig(
                alert_type=AlertType.DENSITY_ALERT,
                alert_level=AlertLevel.DANGER,
                threshold_value=6.0,
                cooldown_seconds=30,
                message_template="High density detected: {density:.2f} people/m² with {people_count} people"
            ),
            AlertConfig(
                alert_type=AlertType.DENSITY_ALERT,
                alert_level=AlertLevel.WARNING,
                threshold_value=4.0,
                cooldown_seconds=60,
                message_template="Crowded conditions: {density:.2f} people/m² with {people_count} people"
            ),
            AlertConfig(
                alert_type=AlertType.MOVEMENT_ALERT,
                alert_level=AlertLevel.DANGER,
                threshold_value=0.7,
                cooldown_seconds=45,
                message_template="High movement risk detected: {movement_risk_level} (score: {movement_risk_score:.2f})"
            ),
            AlertConfig(
                alert_type=AlertType.CROWD_FLOW_ALERT,
                alert_level=AlertLevel.WARNING,
                threshold_value=0.5,
                cooldown_seconds=60,
                message_template="Unusual crowd flow: {movement_direction} (intensity: {flow_intensity:.2f})"
            )
        ]
        
        for config in configs:
            self.alert_configs[config.alert_type] = config
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for new alerts"""
        self.alert_callbacks.append(callback)
    
    def configure_email(self, smtp_server: str, smtp_port: int, 
                       username: str, password: str):
        """Configure email notifications"""
        self.email_manager = EmailManager(smtp_server, smtp_port, username, password)
    
    def add_webhook(self, url: str, headers: Optional[Dict[str, str]] = None):
        """Add webhook for notifications"""
        self.webhook_manager.add_webhook(url, headers)
    
    def check_density_alert(self, camera_id: int, density: float, people_count: int) -> Optional[Alert]:
        """Check for density-based alerts"""
        config = self.alert_configs.get(AlertType.DENSITY_ALERT)
        if not config:
            return None
        
        # Check cooldown
        alert_key = f"density_{camera_id}"
        if alert_key in self.last_alert_times:
            if time.time() - self.last_alert_times[alert_key] < config.cooldown_seconds:
                return None
        
        # Check threshold
        if density >= config.threshold_value:
            alert_level = AlertLevel.DANGER if density >= 6.0 else AlertLevel.WARNING
            
            message = config.message_template.format(
                density=density,
                people_count=people_count
            )
            
            alert = self._create_alert(
                camera_id=camera_id,
                alert_type=AlertType.DENSITY_ALERT,
                alert_level=alert_level,
                message=message,
                data={
                    'density': density,
                    'people_count': people_count,
                    'threshold': config.threshold_value
                }
            )
            
            self.last_alert_times[alert_key] = time.time()
            return alert
        
        return None
    
    def check_movement_alert(self, camera_id: int, movement_risk_score: float, 
                           movement_risk_level: str) -> Optional[Alert]:
        """Check for movement-based alerts"""
        config = self.alert_configs.get(AlertType.MOVEMENT_ALERT)
        if not config:
            return None
        
        # Check cooldown
        alert_key = f"movement_{camera_id}"
        if alert_key in self.last_alert_times:
            if time.time() - self.last_alert_times[alert_key] < config.cooldown_seconds:
                return None
        
        # Check threshold
        if movement_risk_score >= config.threshold_value:
            alert_level = AlertLevel.DANGER if movement_risk_score >= 0.8 else AlertLevel.WARNING
            
            message = config.message_template.format(
                movement_risk_level=movement_risk_level,
                movement_risk_score=movement_risk_score
            )
            
            alert = self._create_alert(
                camera_id=camera_id,
                alert_type=AlertType.MOVEMENT_ALERT,
                alert_level=alert_level,
                message=message,
                data={
                    'movement_risk_score': movement_risk_score,
                    'movement_risk_level': movement_risk_level,
                    'threshold': config.threshold_value
                }
            )
            
            self.last_alert_times[alert_key] = time.time()
            return alert
        
        return None
    
    def check_crowd_flow_alert(self, camera_id: int, flow_intensity: float, 
                             movement_direction: str) -> Optional[Alert]:
        """Check for crowd flow alerts"""
        config = self.alert_configs.get(AlertType.CROWD_FLOW_ALERT)
        if not config:
            return None
        
        # Check cooldown
        alert_key = f"flow_{camera_id}"
        if alert_key in self.last_alert_times:
            if time.time() - self.last_alert_times[alert_key] < config.cooldown_seconds:
                return None
        
        # Check threshold
        if flow_intensity >= config.threshold_value:
            alert_level = AlertLevel.WARNING
            
            message = config.message_template.format(
                movement_direction=movement_direction,
                flow_intensity=flow_intensity
            )
            
            alert = self._create_alert(
                camera_id=camera_id,
                alert_type=AlertType.CROWD_FLOW_ALERT,
                alert_level=alert_level,
                message=message,
                data={
                    'flow_intensity': flow_intensity,
                    'movement_direction': movement_direction,
                    'threshold': config.threshold_value
                }
            )
            
            self.last_alert_times[alert_key] = time.time()
            return alert
        
        return None
    
    def create_custom_alert(self, camera_id: int, alert_level: AlertLevel, 
                          message: str, data: Optional[Dict[str, Any]] = None) -> Alert:
        """Create a custom alert"""
        return self._create_alert(
            camera_id=camera_id,
            alert_type=AlertType.CUSTOM_ALERT,
            alert_level=alert_level,
            message=message,
            data=data or {}
        )
    
    def _create_alert(self, camera_id: int, alert_type: AlertType, 
                     alert_level: AlertLevel, message: str, data: Dict[str, Any]) -> Alert:
        """Create a new alert"""
        alert_id = f"{alert_type.value}_{camera_id}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            timestamp=time.time(),
            camera_id=camera_id,
            alert_type=alert_type,
            alert_level=alert_level,
            message=message,
            data=data
        )
        
        # Store alert
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Keep history size manageable
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        # Trigger notifications
        self._trigger_notifications(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"[AlertManager] Callback error: {e}")
        
        print(f"[AlertManager] Alert created: {alert_level.value} - {message}")
        return alert
    
    def _trigger_notifications(self, alert: Alert):
        """Trigger all configured notifications"""
        # Sound notification
        if self.sound_manager.enabled:
            self.sound_manager.play_alert(alert.alert_level)
        
        # Email notification
        if self.email_manager.enabled:
            # You would need to configure recipients
            recipients = []  # Add email recipients here
            if recipients:
                self.email_manager.send_alert(alert, recipients)
        
        # Webhook notification
        if self.webhook_manager.enabled:
            self.webhook_manager.send_alert(alert)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = time.time()
            print(f"[AlertManager] Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        return False
    
    def get_active_alerts(self, camera_id: Optional[int] = None) -> List[Alert]:
        """Get active (unacknowledged) alerts"""
        alerts = []
        for alert in self.alert_history:
            if not alert.acknowledged:
                if camera_id is None or alert.camera_id == camera_id:
                    alerts.append(alert)
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_history(self, camera_id: Optional[int] = None, 
                         limit: int = 100) -> List[Alert]:
        """Get alert history"""
        alerts = []
        for alert in self.alert_history:
            if camera_id is None or alert.camera_id == camera_id:
                alerts.append(alert)
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_alert_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics for the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert.timestamp >= cutoff_time
        ]
        
        stats = {
            'total_alerts': len(recent_alerts),
            'by_level': {},
            'by_type': {},
            'by_camera': {},
            'acknowledged': 0,
            'unacknowledged': 0
        }
        
        for alert in recent_alerts:
            # By level
            level = alert.alert_level.value
            stats['by_level'][level] = stats['by_level'].get(level, 0) + 1
            
            # By type
            alert_type = alert.alert_type.value
            stats['by_type'][alert_type] = stats['by_type'].get(alert_type, 0) + 1
            
            # By camera
            camera_id = alert.camera_id
            stats['by_camera'][camera_id] = stats['by_camera'].get(camera_id, 0) + 1
            
            # Acknowledged status
            if alert.acknowledged:
                stats['acknowledged'] += 1
            else:
                stats['unacknowledged'] += 1
        
        return stats
    
    def cleanup_old_alerts(self, days_to_keep: int = 7):
        """Clean up old alerts"""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        # Remove from active alerts
        to_remove = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.timestamp < cutoff_time
        ]
        
        for alert_id in to_remove:
            del self.alerts[alert_id]
        
        # Remove from history
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
        
        print(f"[AlertManager] Cleaned up {len(to_remove)} old alerts")
        return len(to_remove)
