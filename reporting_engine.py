"""
Reporting Engine for STAMPede Detection System
Generates comprehensive reports and analytics from historical data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import json
import os
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO

class ReportType(Enum):
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_ANALYSIS = "weekly_analysis"
    MONTHLY_REPORT = "monthly_report"
    CUSTOM_PERIOD = "custom_period"
    INCIDENT_REPORT = "incident_report"
    PERFORMANCE_REPORT = "performance_report"
    TREND_ANALYSIS = "trend_analysis"

class ChartType(Enum):
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX = "box"

@dataclass
class ReportConfig:
    report_type: ReportType
    start_time: float
    end_time: float
    camera_ids: List[int]
    include_charts: bool = True
    chart_format: str = "png"  # png, svg, html
    include_raw_data: bool = False
    email_recipients: List[str] = None
    output_format: str = "pdf"  # pdf, html, json, csv

@dataclass
class ReportResult:
    report_id: str
    report_type: ReportType
    generated_at: float
    file_path: str
    summary: Dict[str, Any]
    charts: List[str]  # Base64 encoded chart images
    raw_data: Optional[Dict[str, Any]] = None

class ReportingEngine:
    """Generates comprehensive reports and analytics"""
    
    def __init__(self, db_manager, output_dir: str = "reports"):
        self.db_manager = db_manager
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_daily_summary(self, date: datetime, camera_ids: List[int] = None) -> ReportResult:
        """Generate daily summary report"""
        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        end_time = date.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp()
        
        config = ReportConfig(
            report_type=ReportType.DAILY_SUMMARY,
            start_time=start_time,
            end_time=end_time,
            camera_ids=camera_ids or []
        )
        
        return self._generate_report(config)
    
    def generate_weekly_analysis(self, week_start: datetime, camera_ids: List[int] = None) -> ReportResult:
        """Generate weekly analysis report"""
        start_time = week_start.timestamp()
        end_time = (week_start + timedelta(days=7)).timestamp()
        
        config = ReportConfig(
            report_type=ReportType.WEEKLY_ANALYSIS,
            start_time=start_time,
            end_time=end_time,
            camera_ids=camera_ids or []
        )
        
        return self._generate_report(config)
    
    def generate_monthly_report(self, month: int, year: int, camera_ids: List[int] = None) -> ReportResult:
        """Generate monthly report"""
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)
        
        config = ReportConfig(
            report_type=ReportType.MONTHLY_REPORT,
            start_time=start_date.timestamp(),
            end_time=end_date.timestamp(),
            camera_ids=camera_ids or []
        )
        
        return self._generate_report(config)
    
    def generate_custom_report(self, start_time: float, end_time: float, 
                             camera_ids: List[int] = None) -> ReportResult:
        """Generate custom period report"""
        config = ReportConfig(
            report_type=ReportType.CUSTOM_PERIOD,
            start_time=start_time,
            end_time=end_time,
            camera_ids=camera_ids or []
        )
        
        return self._generate_report(config)
    
    def _generate_report(self, config: ReportConfig) -> ReportResult:
        """Generate report based on configuration"""
        report_id = f"{config.report_type.value}_{int(time.time())}"
        
        # Get data
        detection_data = self._get_detection_data(config)
        alert_data = self._get_alert_data(config)
        
        # Generate summary
        summary = self._generate_summary(detection_data, alert_data, config)
        
        # Generate charts
        charts = []
        if config.include_charts:
            charts = self._generate_charts(detection_data, alert_data, config)
        
        # Generate raw data if requested
        raw_data = None
        if config.include_raw_data:
            raw_data = {
                'detections': detection_data.to_dict('records') if not detection_data.empty else [],
                'alerts': alert_data.to_dict('records') if not alert_data.empty else []
            }
        
        # Save report
        file_path = self._save_report(report_id, summary, charts, raw_data, config)
        
        return ReportResult(
            report_id=report_id,
            report_type=config.report_type,
            generated_at=time.time(),
            file_path=file_path,
            summary=summary,
            charts=charts,
            raw_data=raw_data
        )
    
    def _get_detection_data(self, config: ReportConfig) -> pd.DataFrame:
        """Get detection data for the specified period"""
        records = self.db_manager.get_detection_records(
            camera_id=None if not config.camera_ids else config.camera_ids[0],
            start_time=config.start_time,
            end_time=config.end_time,
            limit=50000
        )
        
        if not records:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for record in records:
            data.append({
                'timestamp': record.timestamp,
                'camera_id': record.camera_id,
                'people_count': record.people_count,
                'density': record.density,
                'max_density': record.max_density,
                'avg_density': record.avg_density,
                'status': record.status,
                'alert_level': record.alert_level,
                'risk_score': record.risk_score,
                'risk_level': record.risk_level,
                'flow_intensity': record.flow_intensity,
                'movement_direction': record.movement_direction,
                'movement_risk_score': record.movement_risk_score,
                'movement_risk_level': record.movement_risk_level,
                'area_m2': record.area_m2
            })
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.day_name()
            df['date'] = df['datetime'].dt.date
        
        return df
    
    def _get_alert_data(self, config: ReportConfig) -> pd.DataFrame:
        """Get alert data for the specified period"""
        records = self.db_manager.get_alert_records(
            camera_id=None if not config.camera_ids else config.camera_ids[0],
            start_time=config.start_time,
            end_time=config.end_time,
            limit=10000
        )
        
        if not records:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for record in records:
            data.append({
                'timestamp': record.timestamp,
                'camera_id': record.camera_id,
                'alert_type': record.alert_type,
                'alert_level': record.alert_level,
                'message': record.message,
                'people_count': record.people_count,
                'density': record.density,
                'risk_score': record.risk_score,
                'acknowledged': record.acknowledged,
                'acknowledged_by': record.acknowledged_by
            })
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.day_name()
            df['date'] = df['datetime'].dt.date
        
        return df
    
    def _generate_summary(self, detection_data: pd.DataFrame, 
                         alert_data: pd.DataFrame, config: ReportConfig) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {
            'report_type': config.report_type.value,
            'period': {
                'start': datetime.fromtimestamp(config.start_time).isoformat(),
                'end': datetime.fromtimestamp(config.end_time).isoformat(),
                'duration_hours': (config.end_time - config.start_time) / 3600
            },
            'cameras': config.camera_ids,
            'detection_summary': {},
            'alert_summary': {},
            'trends': {},
            'insights': []
        }
        
        # Detection summary
        if not detection_data.empty:
            summary['detection_summary'] = {
                'total_records': len(detection_data),
                'avg_people_count': float(detection_data['people_count'].mean()),
                'max_people_count': int(detection_data['people_count'].max()),
                'avg_density': float(detection_data['density'].mean()),
                'max_density': float(detection_data['density'].max()),
                'avg_risk_score': float(detection_data['risk_score'].mean()),
                'max_risk_score': float(detection_data['risk_score'].max()),
                'status_distribution': detection_data['status'].value_counts().to_dict(),
                'alert_level_distribution': detection_data['alert_level'].value_counts().to_dict(),
                'risk_level_distribution': detection_data['risk_level'].value_counts().to_dict()
            }
            
            # Hourly patterns
            if 'hour' in detection_data.columns:
                hourly_stats = detection_data.groupby('hour').agg({
                    'people_count': ['mean', 'max'],
                    'density': ['mean', 'max'],
                    'risk_score': 'mean'
                }).round(2)
                summary['detection_summary']['hourly_patterns'] = hourly_stats.to_dict()
            
            # Daily patterns
            if 'day_of_week' in detection_data.columns:
                daily_stats = detection_data.groupby('day_of_week').agg({
                    'people_count': ['mean', 'max'],
                    'density': ['mean', 'max'],
                    'risk_score': 'mean'
                }).round(2)
                summary['detection_summary']['daily_patterns'] = daily_stats.to_dict()
        
        # Alert summary
        if not alert_data.empty:
            summary['alert_summary'] = {
                'total_alerts': len(alert_data),
                'acknowledged_alerts': int(alert_data['acknowledged'].sum()),
                'unacknowledged_alerts': int((~alert_data['acknowledged']).sum()),
                'alert_type_distribution': alert_data['alert_type'].value_counts().to_dict(),
                'alert_level_distribution': alert_data['alert_level'].value_counts().to_dict(),
                'avg_people_count': float(alert_data['people_count'].mean()),
                'avg_density': float(alert_data['density'].mean()),
                'avg_risk_score': float(alert_data['risk_score'].mean())
            }
            
            # Alert trends
            if 'hour' in alert_data.columns:
                hourly_alerts = alert_data.groupby('hour').size()
                summary['alert_summary']['hourly_alert_patterns'] = hourly_alerts.to_dict()
        
        # Trends analysis
        if not detection_data.empty and len(detection_data) > 10:
            summary['trends'] = self._analyze_trends(detection_data)
        
        # Generate insights
        summary['insights'] = self._generate_insights(detection_data, alert_data)
        
        return summary
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in the data"""
        trends = {}
        
        if 'timestamp' in data.columns and 'density' in data.columns:
            # Sort by timestamp
            data_sorted = data.sort_values('timestamp')
            
            # Calculate density trend
            if len(data_sorted) > 1:
                x = data_sorted['timestamp'].values
                y = data_sorted['density'].values
                
                # Linear regression
                coeffs = np.polyfit(x, y, 1)
                slope = coeffs[0]
                
                trends['density_trend'] = {
                    'direction': 'increasing' if slope > 0 else 'decreasing',
                    'slope': float(slope),
                    'strength': min(1.0, abs(slope) * 1000)
                }
            
            # Calculate people count trend
            if 'people_count' in data_sorted.columns:
                y_people = data_sorted['people_count'].values
                coeffs_people = np.polyfit(x, y_people, 1)
                slope_people = coeffs_people[0]
                
                trends['people_count_trend'] = {
                    'direction': 'increasing' if slope_people > 0 else 'decreasing',
                    'slope': float(slope_people),
                    'strength': min(1.0, abs(slope_people) * 100)
                }
        
        return trends
    
    def _generate_insights(self, detection_data: pd.DataFrame, 
                          alert_data: pd.DataFrame) -> List[str]:
        """Generate insights from the data"""
        insights = []
        
        if detection_data.empty:
            insights.append("No detection data available for the specified period.")
            return insights
        
        # Density insights
        max_density = detection_data['density'].max()
        avg_density = detection_data['density'].mean()
        
        if max_density > 6.0:
            insights.append(f"High density detected: Maximum density reached {max_density:.2f} people/m²")
        elif max_density > 4.0:
            insights.append(f"Moderate crowding: Maximum density reached {max_density:.2f} people/m²")
        else:
            insights.append(f"Low density conditions: Maximum density was {max_density:.2f} people/m²")
        
        # People count insights
        max_people = detection_data['people_count'].max()
        avg_people = detection_data['people_count'].mean()
        
        if max_people > 20:
            insights.append(f"Large crowds detected: Maximum {max_people} people observed")
        elif max_people > 10:
            insights.append(f"Medium crowds detected: Maximum {max_people} people observed")
        else:
            insights.append(f"Small crowds: Maximum {max_people} people observed")
        
        # Risk insights
        if 'risk_score' in detection_data.columns:
            max_risk = detection_data['risk_score'].max()
            if max_risk > 0.7:
                insights.append(f"High risk periods detected: Maximum risk score {max_risk:.2f}")
            elif max_risk > 0.4:
                insights.append(f"Moderate risk periods: Maximum risk score {max_risk:.2f}")
        
        # Alert insights
        if not alert_data.empty:
            total_alerts = len(alert_data)
            unacknowledged = (~alert_data['acknowledged']).sum()
            
            insights.append(f"Alert activity: {total_alerts} alerts generated")
            if unacknowledged > 0:
                insights.append(f"Attention needed: {unacknowledged} unacknowledged alerts")
        
        # Time-based insights
        if 'hour' in detection_data.columns:
            peak_hour = detection_data.groupby('hour')['people_count'].mean().idxmax()
            insights.append(f"Peak activity hour: {peak_hour}:00")
        
        return insights
    
    def _generate_charts(self, detection_data: pd.DataFrame, 
                        alert_data: pd.DataFrame, config: ReportConfig) -> List[str]:
        """Generate charts and return as base64 encoded images"""
        charts = []
        
        if detection_data.empty:
            return charts
        
        # 1. People count over time
        if 'timestamp' in detection_data.columns and 'people_count' in detection_data.columns:
            chart = self._create_people_count_chart(detection_data)
            if chart:
                charts.append(chart)
        
        # 2. Density over time
        if 'timestamp' in detection_data.columns and 'density' in detection_data.columns:
            chart = self._create_density_chart(detection_data)
            if chart:
                charts.append(chart)
        
        # 3. Hourly patterns
        if 'hour' in detection_data.columns:
            chart = self._create_hourly_pattern_chart(detection_data)
            if chart:
                charts.append(chart)
        
        # 4. Status distribution
        if 'status' in detection_data.columns:
            chart = self._create_status_distribution_chart(detection_data)
            if chart:
                charts.append(chart)
        
        # 5. Alert timeline
        if not alert_data.empty and 'timestamp' in alert_data.columns:
            chart = self._create_alert_timeline_chart(alert_data)
            if chart:
                charts.append(chart)
        
        # 6. Risk score distribution
        if 'risk_score' in detection_data.columns:
            chart = self._create_risk_distribution_chart(detection_data)
            if chart:
                charts.append(chart)
        
        return charts
    
    def _create_people_count_chart(self, data: pd.DataFrame) -> Optional[str]:
        """Create people count over time chart"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Sample data if too many points
            if len(data) > 1000:
                data_sampled = data.sample(1000).sort_values('timestamp')
            else:
                data_sampled = data.sort_values('timestamp')
            
            ax.plot(data_sampled['timestamp'], data_sampled['people_count'], 
                   linewidth=1, alpha=0.7, color='blue')
            ax.set_title('People Count Over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('People Count')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.tick_params(axis='x', rotation=45)
            
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"[ReportingEngine] Error creating people count chart: {e}")
            return None
    
    def _create_density_chart(self, data: pd.DataFrame) -> Optional[str]:
        """Create density over time chart"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Sample data if too many points
            if len(data) > 1000:
                data_sampled = data.sample(1000).sort_values('timestamp')
            else:
                data_sampled = data.sort_values('timestamp')
            
            ax.plot(data_sampled['timestamp'], data_sampled['density'], 
                   linewidth=1, alpha=0.7, color='red')
            
            # Add threshold lines
            ax.axhline(y=4.0, color='orange', linestyle='--', alpha=0.7, label='Warning (4 people/m²)')
            ax.axhline(y=6.0, color='red', linestyle='--', alpha=0.7, label='Danger (6 people/m²)')
            
            ax.set_title('Density Over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('Density (people/m²)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.tick_params(axis='x', rotation=45)
            
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"[ReportingEngine] Error creating density chart: {e}")
            return None
    
    def _create_hourly_pattern_chart(self, data: pd.DataFrame) -> Optional[str]:
        """Create hourly pattern chart"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # People count by hour
            hourly_people = data.groupby('hour')['people_count'].mean()
            ax1.bar(hourly_people.index, hourly_people.values, alpha=0.7, color='blue')
            ax1.set_title('Average People Count by Hour')
            ax1.set_xlabel('Hour')
            ax1.set_ylabel('Average People Count')
            ax1.grid(True, alpha=0.3)
            
            # Density by hour
            hourly_density = data.groupby('hour')['density'].mean()
            ax2.bar(hourly_density.index, hourly_density.values, alpha=0.7, color='red')
            ax2.set_title('Average Density by Hour')
            ax2.set_xlabel('Hour')
            ax2.set_ylabel('Average Density (people/m²)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"[ReportingEngine] Error creating hourly pattern chart: {e}")
            return None
    
    def _create_status_distribution_chart(self, data: pd.DataFrame) -> Optional[str]:
        """Create status distribution pie chart"""
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            status_counts = data['status'].value_counts()
            colors = ['green', 'yellow', 'red', 'orange']
            
            ax.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                  colors=colors[:len(status_counts)], startangle=90)
            ax.set_title('Status Distribution')
            
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"[ReportingEngine] Error creating status distribution chart: {e}")
            return None
    
    def _create_alert_timeline_chart(self, data: pd.DataFrame) -> Optional[str]:
        """Create alert timeline chart"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Count alerts by hour
            data['hour'] = pd.to_datetime(data['timestamp'], unit='s').dt.hour
            hourly_alerts = data.groupby('hour').size()
            
            ax.bar(hourly_alerts.index, hourly_alerts.values, alpha=0.7, color='red')
            ax.set_title('Alert Timeline by Hour')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Number of Alerts')
            ax.grid(True, alpha=0.3)
            
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"[ReportingEngine] Error creating alert timeline chart: {e}")
            return None
    
    def _create_risk_distribution_chart(self, data: pd.DataFrame) -> Optional[str]:
        """Create risk score distribution histogram"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(data['risk_score'], bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax.set_title('Risk Score Distribution')
            ax.set_xlabel('Risk Score')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"[ReportingEngine] Error creating risk distribution chart: {e}")
            return None
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return image_base64
    
    def _save_report(self, report_id: str, summary: Dict[str, Any], 
                    charts: List[str], raw_data: Optional[Dict[str, Any]], 
                    config: ReportConfig) -> str:
        """Save report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_id}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        report_data = {
            'report_id': report_id,
            'generated_at': time.time(),
            'config': {
                'report_type': config.report_type.value,
                'start_time': config.start_time,
                'end_time': config.end_time,
                'camera_ids': config.camera_ids
            },
            'summary': summary,
            'charts': charts,
            'raw_data': raw_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"[ReportingEngine] Report saved: {filepath}")
        return filepath
    
    def get_report_list(self) -> List[Dict[str, Any]]:
        """Get list of available reports"""
        reports = []
        
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.output_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        report_data = json.load(f)
                    
                    reports.append({
                        'report_id': report_data['report_id'],
                        'generated_at': report_data['generated_at'],
                        'report_type': report_data['config']['report_type'],
                        'file_path': filepath,
                        'summary': report_data['summary']
                    })
                except Exception as e:
                    print(f"[ReportingEngine] Error reading report {filename}: {e}")
        
        return sorted(reports, key=lambda x: x['generated_at'], reverse=True)
    
    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get specific report by ID"""
        for filename in os.listdir(self.output_dir):
            if filename.startswith(report_id):
                filepath = os.path.join(self.output_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"[ReportingEngine] Error reading report {filename}: {e}")
        
        return None
