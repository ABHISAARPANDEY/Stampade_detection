"""
Authentication and Authorization Manager for STAMPede Detection System
Implements user authentication, role-based access control, and session management
"""

import hashlib
import secrets
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from datetime import datetime, timedelta
import jwt
import bcrypt
from functools import wraps

class UserRole(Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    GUEST = "guest"

class Permission(Enum):
    VIEW_DETECTIONS = "view_detections"
    CREATE_DETECTIONS = "create_detections"
    VIEW_ALERTS = "view_alerts"
    ACKNOWLEDGE_ALERTS = "acknowledge_alerts"
    MANAGE_CAMERAS = "manage_cameras"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_USERS = "manage_users"
    SYSTEM_SETTINGS = "system_settings"
    EXPORT_DATA = "export_data"
    API_ACCESS = "api_access"

@dataclass
class User:
    id: Optional[int] = None
    username: str = ""
    email: str = ""
    password_hash: str = ""
    role: UserRole = UserRole.GUEST
    is_active: bool = True
    created_at: float = 0.0
    last_login: Optional[float] = None
    failed_login_attempts: int = 0
    locked_until: Optional[float] = None
    preferences: str = "{}"  # JSON string

@dataclass
class Session:
    session_id: str
    user_id: int
    created_at: float
    expires_at: float
    ip_address: str
    user_agent: str
    is_active: bool = True

@dataclass
class RolePermission:
    role: UserRole
    permissions: List[Permission]

class AuthManager:
    """Manages user authentication and authorization"""
    
    def __init__(self, db_path: str = "auth.db", jwt_secret: str = None):
        self.db_path = db_path
        self.jwt_secret = jwt_secret or secrets.token_hex(32)
        self.session_timeout = 3600  # 1 hour
        self.max_failed_attempts = 5
        self.lockout_duration = 900  # 15 minutes
        
        # Role permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: [
                Permission.VIEW_DETECTIONS,
                Permission.CREATE_DETECTIONS,
                Permission.VIEW_ALERTS,
                Permission.ACKNOWLEDGE_ALERTS,
                Permission.MANAGE_CAMERAS,
                Permission.VIEW_ANALYTICS,
                Permission.MANAGE_USERS,
                Permission.SYSTEM_SETTINGS,
                Permission.EXPORT_DATA,
                Permission.API_ACCESS
            ],
            UserRole.OPERATOR: [
                Permission.VIEW_DETECTIONS,
                Permission.CREATE_DETECTIONS,
                Permission.VIEW_ALERTS,
                Permission.ACKNOWLEDGE_ALERTS,
                Permission.MANAGE_CAMERAS,
                Permission.VIEW_ANALYTICS,
                Permission.EXPORT_DATA
            ],
            UserRole.VIEWER: [
                Permission.VIEW_DETECTIONS,
                Permission.VIEW_ALERTS,
                Permission.VIEW_ANALYTICS
            ],
            UserRole.GUEST: [
                Permission.VIEW_DETECTIONS
            ]
        }
        
        # Active sessions
        self.active_sessions: Dict[str, Session] = {}
        
        # Initialize database
        self._init_database()
        
        # Create default admin user if none exists
        self._create_default_admin()
    
    def _init_database(self):
        """Initialize authentication database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at REAL NOT NULL,
                    last_login REAL,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until REAL,
                    preferences TEXT DEFAULT '{}'
                )
            """)
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Audit log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    resource TEXT,
                    details TEXT,
                    ip_address TEXT,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            conn.commit()
    
    def _create_default_admin(self):
        """Create default admin user if none exists"""
        if not self.get_user_by_username("admin"):
            self.create_user(
                username="admin",
                email="admin@stampede.local",
                password="admin123",  # Should be changed in production
                role=UserRole.ADMIN
            )
            print("[AuthManager] Created default admin user (username: admin, password: admin123)")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return secrets.token_urlsafe(32)
    
    def _log_audit_event(self, user_id: Optional[int], action: str, 
                        resource: str = "", details: str = "", 
                        ip_address: str = ""):
        """Log audit event"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_log (user_id, action, resource, details, ip_address, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, action, resource, details, ip_address, time.time()))
            conn.commit()
    
    def create_user(self, username: str, email: str, password: str, 
                   role: UserRole = UserRole.GUEST) -> Optional[User]:
        """Create a new user"""
        try:
            # Check if user already exists
            if self.get_user_by_username(username):
                return None
            
            if self.get_user_by_email(email):
                return None
            
            # Hash password
            password_hash = self._hash_password(password)
            
            # Create user
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                role=role,
                created_at=time.time()
            )
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (username, email, password_hash, role, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (user.username, user.email, user.password_hash, user.role.value, user.created_at))
                
                user.id = cursor.lastrowid
                conn.commit()
            
            self._log_audit_event(user.id, "user_created", "user", f"Created user {username}")
            print(f"[AuthManager] Created user: {username} ({role.value})")
            return user
        
        except Exception as e:
            print(f"[AuthManager] Failed to create user: {e}")
            return None
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str = "", user_agent: str = "") -> Optional[Session]:
        """Authenticate user and create session"""
        try:
            user = self.get_user_by_username(username)
            if not user:
                self._log_audit_event(None, "login_failed", "user", f"Invalid username: {username}", ip_address)
                return None
            
            # Check if account is locked
            if user.locked_until and time.time() < user.locked_until:
                self._log_audit_event(user.id, "login_failed", "user", "Account locked", ip_address)
                return None
            
            # Check if account is active
            if not user.is_active:
                self._log_audit_event(user.id, "login_failed", "user", "Account inactive", ip_address)
                return None
            
            # Verify password
            if not self._verify_password(password, user.password_hash):
                # Increment failed attempts
                self._increment_failed_attempts(user.id)
                self._log_audit_event(user.id, "login_failed", "user", "Invalid password", ip_address)
                return None
            
            # Reset failed attempts on successful login
            self._reset_failed_attempts(user.id)
            
            # Update last login
            self._update_last_login(user.id)
            
            # Create session
            session = self._create_session(user.id, ip_address, user_agent)
            
            self._log_audit_event(user.id, "login_success", "user", f"User {username} logged in", ip_address)
            print(f"[AuthManager] User {username} authenticated successfully")
            return session
        
        except Exception as e:
            print(f"[AuthManager] Authentication failed: {e}")
            return None
    
    def _create_session(self, user_id: int, ip_address: str, user_agent: str) -> Session:
        """Create a new session"""
        session_id = self._generate_session_id()
        now = time.time()
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            expires_at=now + self.session_timeout,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (session_id, user_id, created_at, expires_at, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session.session_id, session.user_id, session.created_at, 
                  session.expires_at, session.ip_address, session.user_agent))
            conn.commit()
        
        # Add to active sessions
        self.active_sessions[session_id] = session
        
        return session
    
    def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate session and return session if valid"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # Check if session is expired
        if time.time() > session.expires_at:
            self.logout_session(session_id)
            return None
        
        # Extend session if it's close to expiring
        if time.time() > session.expires_at - 300:  # 5 minutes before expiry
            self._extend_session(session_id)
        
        return session
    
    def _extend_session(self, session_id: str):
        """Extend session expiry time"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.expires_at = time.time() + self.session_timeout
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE sessions SET expires_at = ? WHERE session_id = ?
                """, (session.expires_at, session_id))
                conn.commit()
    
    def logout_session(self, session_id: str) -> bool:
        """Logout and invalidate session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Mark as inactive in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE sessions SET is_active = FALSE WHERE session_id = ?
                """, (session_id,))
                conn.commit()
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            self._log_audit_event(session.user_id, "logout", "session", f"Session {session_id} ended")
            return True
        
        return False
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            
            if row:
                return User(
                    id=row[0],
                    username=row[1],
                    email=row[2],
                    password_hash=row[3],
                    role=UserRole(row[4]),
                    is_active=bool(row[5]),
                    created_at=row[6],
                    last_login=row[7],
                    failed_login_attempts=row[8],
                    locked_until=row[9],
                    preferences=row[10]
                )
        
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = cursor.fetchone()
            
            if row:
                return User(
                    id=row[0],
                    username=row[1],
                    email=row[2],
                    password_hash=row[3],
                    role=UserRole(row[4]),
                    is_active=bool(row[5]),
                    created_at=row[6],
                    last_login=row[7],
                    failed_login_attempts=row[8],
                    locked_until=row[9],
                    preferences=row[10]
                )
        
        return None
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            
            if row:
                return User(
                    id=row[0],
                    username=row[1],
                    email=row[2],
                    password_hash=row[3],
                    role=UserRole(row[4]),
                    is_active=bool(row[5]),
                    created_at=row[6],
                    last_login=row[7],
                    failed_login_attempts=row[8],
                    locked_until=row[9],
                    preferences=row[10]
                )
        
        return None
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission"""
        if not user.is_active:
            return False
        
        user_permissions = self.role_permissions.get(user.role, [])
        return permission in user_permissions
    
    def has_any_permission(self, user: User, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions"""
        if not user.is_active:
            return False
        
        user_permissions = self.role_permissions.get(user.role, [])
        return any(permission in user_permissions for permission in permissions)
    
    def has_all_permissions(self, user: User, permissions: List[Permission]) -> bool:
        """Check if user has all of the specified permissions"""
        if not user.is_active:
            return False
        
        user_permissions = self.role_permissions.get(user.role, [])
        return all(permission in user_permissions for permission in permissions)
    
    def get_user_permissions(self, user: User) -> List[Permission]:
        """Get all permissions for a user"""
        if not user.is_active:
            return []
        
        return self.role_permissions.get(user.role, [])
    
    def update_user_role(self, user_id: int, new_role: UserRole) -> bool:
        """Update user role"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users SET role = ? WHERE id = ?
                """, (new_role.value, user_id))
                conn.commit()
            
            self._log_audit_event(user_id, "role_updated", "user", f"Role changed to {new_role.value}")
            return True
        except Exception as e:
            print(f"[AuthManager] Failed to update role: {e}")
            return False
    
    def update_user_password(self, user_id: int, new_password: str) -> bool:
        """Update user password"""
        try:
            password_hash = self._hash_password(new_password)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users SET password_hash = ? WHERE id = ?
                """, (password_hash, user_id))
                conn.commit()
            
            self._log_audit_event(user_id, "password_updated", "user", "Password changed")
            return True
        except Exception as e:
            print(f"[AuthManager] Failed to update password: {e}")
            return False
    
    def deactivate_user(self, user_id: int) -> bool:
        """Deactivate user account"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users SET is_active = FALSE WHERE id = ?
                """, (user_id,))
                conn.commit()
            
            # Logout all sessions for this user
            self._logout_user_sessions(user_id)
            
            self._log_audit_event(user_id, "user_deactivated", "user", "User account deactivated")
            return True
        except Exception as e:
            print(f"[AuthManager] Failed to deactivate user: {e}")
            return False
    
    def _increment_failed_attempts(self, user_id: int):
        """Increment failed login attempts"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET failed_login_attempts = failed_login_attempts + 1 
                WHERE id = ?
            """, (user_id,))
            
            # Check if account should be locked
            cursor.execute("SELECT failed_login_attempts FROM users WHERE id = ?", (user_id,))
            attempts = cursor.fetchone()[0]
            
            if attempts >= self.max_failed_attempts:
                lockout_until = time.time() + self.lockout_duration
                cursor.execute("""
                    UPDATE users SET locked_until = ? WHERE id = ?
                """, (lockout_until, user_id))
            
            conn.commit()
    
    def _reset_failed_attempts(self, user_id: int):
        """Reset failed login attempts"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET failed_login_attempts = 0, locked_until = NULL WHERE id = ?
            """, (user_id,))
            conn.commit()
    
    def _update_last_login(self, user_id: int):
        """Update last login time"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET last_login = ? WHERE id = ?
            """, (time.time(), user_id))
            conn.commit()
    
    def _logout_user_sessions(self, user_id: int):
        """Logout all sessions for a user"""
        sessions_to_remove = []
        for session_id, session in self.active_sessions.items():
            if session.user_id == user_id:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            self.logout_session(session_id)
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if current_time > session.expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.logout_session(session_id)
        
        if expired_sessions:
            print(f"[AuthManager] Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_audit_log(self, user_id: Optional[int] = None, 
                     start_time: Optional[float] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log entries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM audit_log WHERE 1=1"
            params = []
            
            if user_id is not None:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            audit_entries = []
            for row in rows:
                audit_entries.append({
                    'id': row[0],
                    'user_id': row[1],
                    'action': row[2],
                    'resource': row[3],
                    'details': row[4],
                    'ip_address': row[5],
                    'timestamp': row[6]
                })
            
            return audit_entries
    
    def generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': user.id,
            'username': user.username,
            'role': user.role.value,
            'exp': time.time() + self.session_timeout,
            'iat': time.time()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def validate_jwt_token(self, token: str) -> Optional[User]:
        """Validate JWT token and return user"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            user_id = payload.get('user_id')
            
            if user_id:
                return self.get_user_by_id(user_id)
        except jwt.ExpiredSignatureError:
            pass
        except jwt.InvalidTokenError:
            pass
        
        return None
