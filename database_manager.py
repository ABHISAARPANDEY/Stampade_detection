"""
Database Manager for STAMPede Detection System
Handles SQLite database operations for storing detection data and analytics
"""

import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import threading
from contextlib import contextmanager

@dataclass
class DetectionRecord:
    id: Optional[int] = None
    timestamp: float = 0.0
    camera_id: int = 0
    people_count: int = 0
    density: float = 0.0
    max_density: float = 0.0
    avg_density: float = 0.0
    status: str = "SAFE"
    alert_level: str = "safe"
    risk_score: float = 0.0
    risk_level: str = "low"
    flow_intensity: float = 0.0
    movement_direction: str = "stable"
    movement_risk_score: float = 0.0
    movement_risk_level: str = "low"
    detection_boxes: str = "[]"  # JSON string
    confidence_scores: str = "[]"  # JSON string
    risk_factors: str = "[]"  # JSON string
    movement_risk_factors: str = "[]"  # JSON string
    area_m2: float = 25.0
    confidence_threshold: float = 0.20
    grid_w: int = 32
    grid_h: int = 24

@dataclass
class AlertRecord:
    id: Optional[int] = None
    timestamp: float = 0.0
    camera_id: int = 0
    alert_type: str = ""
    alert_level: str = "info"
    message: str = ""
    people_count: int = 0
    density: float = 0.0
    risk_score: float = 0.0
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None

@dataclass
class CameraConfigRecord:
    id: Optional[int] = None
    camera_id: int = 0
    name: str = ""
    resolution_width: int = 1280
    resolution_height: int = 720
    fps: int = 30
    area_m2: float = 25.0
    confidence_threshold: float = 0.20
    grid_w: int = 32
    grid_h: int = 24
    danger_density: float = 6.0
    warning_density: float = 4.0
    enabled: bool = True
    created_at: float = 0.0
    updated_at: float = 0.0

class DatabaseManager:
    """Manages SQLite database operations for the STAMPede Detection System"""
    
    def __init__(self, db_path: str = "stampede_detection.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Detection records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    camera_id INTEGER NOT NULL,
                    people_count INTEGER NOT NULL,
                    density REAL NOT NULL,
                    max_density REAL NOT NULL,
                    avg_density REAL NOT NULL,
                    status TEXT NOT NULL,
                    alert_level TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    flow_intensity REAL NOT NULL,
                    movement_direction TEXT NOT NULL,
                    movement_risk_score REAL NOT NULL,
                    movement_risk_level TEXT NOT NULL,
                    detection_boxes TEXT NOT NULL,
                    confidence_scores TEXT NOT NULL,
                    risk_factors TEXT NOT NULL,
                    movement_risk_factors TEXT NOT NULL,
                    area_m2 REAL NOT NULL,
                    confidence_threshold REAL NOT NULL,
                    grid_w INTEGER NOT NULL,
                    grid_h INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    camera_id INTEGER NOT NULL,
                    alert_type TEXT NOT NULL,
                    alert_level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    people_count INTEGER NOT NULL,
                    density REAL NOT NULL,
                    risk_score REAL NOT NULL,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    acknowledged_by TEXT,
                    acknowledged_at REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Camera configurations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS camera_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id INTEGER UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    resolution_width INTEGER NOT NULL,
                    resolution_height INTEGER NOT NULL,
                    fps INTEGER NOT NULL,
                    area_m2 REAL NOT NULL,
                    confidence_threshold REAL NOT NULL,
                    grid_w INTEGER NOT NULL,
                    grid_h INTEGER NOT NULL,
                    danger_density REAL NOT NULL,
                    warning_density REAL NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            
            # System settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    description TEXT,
                    updated_at REAL NOT NULL
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_timestamp ON detection_records(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_camera ON detection_records(camera_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_camera ON alerts(camera_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged)")
            
            conn.commit()
            print("[Database] Database initialized successfully")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            print(f"[Database] Error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def insert_detection_record(self, record: DetectionRecord) -> int:
        """Insert a new detection record"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO detection_records (
                        timestamp, camera_id, people_count, density, max_density, avg_density,
                        status, alert_level, risk_score, risk_level, flow_intensity,
                        movement_direction, movement_risk_score, movement_risk_level,
                        detection_boxes, confidence_scores, risk_factors, movement_risk_factors,
                        area_m2, confidence_threshold, grid_w, grid_h
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.timestamp, record.camera_id, record.people_count, record.density,
                    record.max_density, record.avg_density, record.status, record.alert_level,
                    record.risk_score, record.risk_level, record.flow_intensity,
                    record.movement_direction, record.movement_risk_score, record.movement_risk_level,
                    record.detection_boxes, record.confidence_scores, record.risk_factors,
                    record.movement_risk_factors, record.area_m2, record.confidence_threshold,
                    record.grid_w, record.grid_h
                ))
                conn.commit()
                return cursor.lastrowid
    
    def insert_alert_record(self, record: AlertRecord) -> int:
        """Insert a new alert record"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO alerts (
                        timestamp, camera_id, alert_type, alert_level, message,
                        people_count, density, risk_score, acknowledged, acknowledged_by, acknowledged_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.timestamp, record.camera_id, record.alert_type, record.alert_level,
                    record.message, record.people_count, record.density, record.risk_score,
                    record.acknowledged, record.acknowledged_by, record.acknowledged_at
                ))
                conn.commit()
                return cursor.lastrowid
    
    def get_detection_records(self, camera_id: Optional[int] = None, 
                            start_time: Optional[float] = None, 
                            end_time: Optional[float] = None,
                            limit: int = 1000) -> List[DetectionRecord]:
        """Get detection records with optional filters"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM detection_records WHERE 1=1"
            params = []
            
            if camera_id is not None:
                query += " AND camera_id = ?"
                params.append(camera_id)
            
            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                record = DetectionRecord(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    camera_id=row['camera_id'],
                    people_count=row['people_count'],
                    density=row['density'],
                    max_density=row['max_density'],
                    avg_density=row['avg_density'],
                    status=row['status'],
                    alert_level=row['alert_level'],
                    risk_score=row['risk_score'],
                    risk_level=row['risk_level'],
                    flow_intensity=row['flow_intensity'],
                    movement_direction=row['movement_direction'],
                    movement_risk_score=row['movement_risk_score'],
                    movement_risk_level=row['movement_risk_level'],
                    detection_boxes=row['detection_boxes'],
                    confidence_scores=row['confidence_scores'],
                    risk_factors=row['risk_factors'],
                    movement_risk_factors=row['movement_risk_factors'],
                    area_m2=row['area_m2'],
                    confidence_threshold=row['confidence_threshold'],
                    grid_w=row['grid_w'],
                    grid_h=row['grid_h']
                )
                records.append(record)
            
            return records
    
    def get_alert_records(self, camera_id: Optional[int] = None,
                         acknowledged: Optional[bool] = None,
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None,
                         limit: int = 1000) -> List[AlertRecord]:
        """Get alert records with optional filters"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM alerts WHERE 1=1"
            params = []
            
            if camera_id is not None:
                query += " AND camera_id = ?"
                params.append(camera_id)
            
            if acknowledged is not None:
                query += " AND acknowledged = ?"
                params.append(acknowledged)
            
            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                record = AlertRecord(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    camera_id=row['camera_id'],
                    alert_type=row['alert_type'],
                    alert_level=row['alert_level'],
                    message=row['message'],
                    people_count=row['people_count'],
                    density=row['density'],
                    risk_score=row['risk_score'],
                    acknowledged=bool(row['acknowledged']),
                    acknowledged_by=row['acknowledged_by'],
                    acknowledged_at=row['acknowledged_at']
                )
                records.append(record)
            
            return records
    
    def acknowledge_alert(self, alert_id: int, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE alerts 
                    SET acknowledged = TRUE, acknowledged_by = ?, acknowledged_at = ?
                    WHERE id = ?
                """, (acknowledged_by, time.time(), alert_id))
                conn.commit()
                return cursor.rowcount > 0
    
    def get_analytics_summary(self, camera_id: Optional[int] = None,
                             start_time: Optional[float] = None,
                             end_time: Optional[float] = None) -> Dict[str, Any]:
        """Get analytics summary for a time period"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Base query
            base_query = "FROM detection_records WHERE 1=1"
            params = []
            
            if camera_id is not None:
                base_query += " AND camera_id = ?"
                params.append(camera_id)
            
            if start_time is not None:
                base_query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time is not None:
                base_query += " AND timestamp <= ?"
                params.append(end_time)
            
            # Get basic statistics
            cursor.execute(f"SELECT COUNT(*) as total_records {base_query}", params)
            total_records = cursor.fetchone()['total_records']
            
            if total_records == 0:
                return {
                    'total_records': 0,
                    'avg_people_count': 0,
                    'avg_density': 0,
                    'max_density': 0,
                    'alert_distribution': {},
                    'risk_level_distribution': {},
                    'hourly_stats': []
                }
            
            # Average statistics
            cursor.execute(f"""
                SELECT 
                    AVG(people_count) as avg_people_count,
                    AVG(density) as avg_density,
                    MAX(max_density) as max_density,
                    AVG(risk_score) as avg_risk_score,
                    AVG(flow_intensity) as avg_flow_intensity
                {base_query}
            """, params)
            stats = cursor.fetchone()
            
            # Alert distribution
            cursor.execute(f"""
                SELECT alert_level, COUNT(*) as count
                {base_query}
                GROUP BY alert_level
            """, params)
            alert_distribution = {row['alert_level']: row['count'] for row in cursor.fetchall()}
            
            # Risk level distribution
            cursor.execute(f"""
                SELECT risk_level, COUNT(*) as count
                {base_query}
                GROUP BY risk_level
            """, params)
            risk_level_distribution = {row['risk_level']: row['count'] for row in cursor.fetchall()}
            
            # Hourly statistics
            cursor.execute(f"""
                SELECT 
                    strftime('%H', datetime(timestamp, 'unixepoch')) as hour,
                    AVG(people_count) as avg_people_count,
                    AVG(density) as avg_density,
                    COUNT(*) as record_count
                {base_query}
                GROUP BY strftime('%H', datetime(timestamp, 'unixepoch'))
                ORDER BY hour
            """, params)
            hourly_stats = [
                {
                    'hour': int(row['hour']),
                    'avg_people_count': row['avg_people_count'],
                    'avg_density': row['avg_density'],
                    'record_count': row['record_count']
                }
                for row in cursor.fetchall()
            ]
            
            return {
                'total_records': total_records,
                'avg_people_count': stats['avg_people_count'] or 0,
                'avg_density': stats['avg_density'] or 0,
                'max_density': stats['max_density'] or 0,
                'avg_risk_score': stats['avg_risk_score'] or 0,
                'avg_flow_intensity': stats['avg_flow_intensity'] or 0,
                'alert_distribution': alert_distribution,
                'risk_level_distribution': risk_level_distribution,
                'hourly_stats': hourly_stats
            }
    
    def cleanup_old_records(self, days_to_keep: int = 30):
        """Clean up old records to save space"""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete old detection records
                cursor.execute("DELETE FROM detection_records WHERE timestamp < ?", (cutoff_time,))
                deleted_detections = cursor.rowcount
                
                # Delete old alerts (keep acknowledged ones longer)
                cursor.execute("""
                    DELETE FROM alerts 
                    WHERE timestamp < ? AND (acknowledged = FALSE OR acknowledged_at < ?)
                """, (cutoff_time, cutoff_time - (7 * 24 * 3600)))  # Keep acknowledged alerts for 7 more days
                deleted_alerts = cursor.rowcount
                
                conn.commit()
                
                print(f"[Database] Cleaned up {deleted_detections} detection records and {deleted_alerts} alerts")
                return deleted_detections + deleted_alerts
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table sizes
            cursor.execute("SELECT COUNT(*) as count FROM detection_records")
            detection_count = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM alerts")
            alert_count = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM camera_configs")
            camera_count = cursor.fetchone()['count']
            
            # Database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()['size']
            
            return {
                'detection_records': detection_count,
                'alerts': alert_count,
                'cameras': camera_count,
                'database_size_bytes': db_size,
                'database_size_mb': db_size / (1024 * 1024)
            }
    
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database"""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            print(f"[Database] Backup created: {backup_path}")
            return True
        except Exception as e:
            print(f"[Database] Backup failed: {e}")
            return False
