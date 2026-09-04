"""
Database Utility for Real-Time Crowd Analysis and Threat Detection
"""

import sqlite3
import os
from datetime import datetime
import threading
import queue
from real_time_crowd_analysis.utils.config import config
from real_time_crowd_analysis.utils.logger import setup_logger

logger = setup_logger("database")


class DatabaseManager:
    """Manages SQLite database operations"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DB_PATH
        self.write_queue = queue.Queue()
        self.init_database()
        
        # Start background writer thread
        self.stop_event = threading.Event()
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
    
    def _writer_loop(self):
        """Background thread for sequential database writes"""
        while not self.stop_event.is_set():
            try:
                # Wait for work
                func, args, callback = self.write_queue.get(timeout=1.0)
                try:
                    result = func(*args)
                    if callback: callback(result)
                except Exception as e:
                    logger.error(f"Async write failed: {e}")
                finally:
                    self.write_queue.task_done()
            except queue.Empty:
                continue

    def _async_write(self, func, args, callback=None):
        """Enqueue a write operation"""
        self.write_queue.put((func, args, callback))

    
    def init_database(self):
        """Initialize database with required tables"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Crowd analytics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS crowd_analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        location_name TEXT,
                        area_name TEXT,
                        camera_id TEXT,
                        camera_type TEXT,
                        network_name TEXT,
                        ip_address TEXT,
                        crowd_count INTEGER,
                        threat_status TEXT,
                        density_level TEXT,
                        session_duration REAL,
                        connection_status TEXT,
                        reconnect_attempts INTEGER
                    )
                ''')

                # Persistent Event History table (Task 2)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS event_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        location TEXT,
                        area TEXT,
                        camera_source TEXT,
                        camera_id TEXT,
                        event_type TEXT,
                        threat_level TEXT,
                        risk_score REAL,
                        motion_score REAL,
                        accel_score REAL,
                        notes TEXT
                    )
                ''')

                # Ensure schema migration if table pre-existed with old 'confidence' column name
                cursor.execute("PRAGMA table_info(event_history)")
                existing_cols = [col[1] for col in cursor.fetchall()]
                if existing_cols and "risk_score" not in existing_cols:
                    try:
                        cursor.execute("ALTER TABLE event_history ADD COLUMN risk_score REAL")
                        logger.info("Migrated event_history table: Added risk_score column")
                    except Exception as migration_err:
                        logger.warning(f"Column migration warning for event_history: {migration_err}")

                # Registered Cameras table (Requirement 5)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cameras (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE,
                        type TEXT,
                        ip_address TEXT,
                        rtsp_url TEXT,
                        location TEXT,
                        area TEXT,
                        status TEXT DEFAULT 'OFFLINE'
                    )
                ''')


                
                # Threat events table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS threat_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        location_name TEXT,
                        area_name TEXT,
                        camera_id TEXT,
                        threat_type TEXT,
                        confidence REAL,
                        frame_saved TEXT,
                        description TEXT
                    )
                ''')
                
                # Network logs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS network_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        camera_type TEXT,
                        network_name TEXT,
                        ip_address TEXT,
                        connection_status TEXT,
                        signal_strength REAL,
                        latency REAL
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def insert_crowd_analytics(self, data: dict):
        """Insert crowd analytics data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO crowd_analytics 
                    (location_name, area_name, camera_id, camera_type, network_name, 
                     ip_address, crowd_count, threat_status, density_level, 
                     session_duration, connection_status, reconnect_attempts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('location_name'),
                    data.get('area_name'),
                    data.get('camera_id'),
                    data.get('camera_type'),
                    data.get('network_name'),
                    data.get('ip_address'),
                    data.get('crowd_count'),
                    data.get('threat_status'),
                    data.get('density_level'),
                    data.get('session_duration'),
                    data.get('connection_status'),
                    data.get('reconnect_attempts', 0)
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to insert crowd analytics: {e}")
            return None
    
    def insert_threat_event(self, data: dict):
        """Insert threat event data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO threat_events 
                    (location_name, area_name, camera_id, threat_type, confidence, 
                     frame_saved, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', ( # Corrected indentation
                    data.get('location_name'),
                    data.get('area_name'),
                    data.get('camera_id'),
                    data.get('threat_type'),
                    data.get('confidence'),
                    data.get('frame_saved'),
                    data.get('description')
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to insert threat event: {e}")
            return None
    
    def insert_network_log(self, data: dict):
        """Insert network log data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO network_logs 
                    (camera_type, network_name, ip_address, connection_status, 
                     signal_strength, latency)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('camera_type'),
                    data.get('network_name'),
                    data.get('ip_address'),
                    data.get('connection_status'),
                    data.get('signal_strength'),
                    data.get('latency')
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to insert network log: {e}")
            return None
            
    def insert_event_history(self, data: dict):
        """Insert a persistent event history record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO event_history 
                    (location, area, camera_source, camera_id, event_type, 
                     threat_level, risk_score, motion_score, accel_score, notes) -- Changed confidence to risk_score
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('location'),
                    data.get('area'),
                    data.get('camera_source'),
                    data.get('camera_id'),
                    data.get('event_type'),
                    data.get('threat_level'),
                    data.get('risk_score'), # Changed confidence to risk_score
                    data.get('motion_score'),
                    data.get('accel_score'),
                    data.get('notes')
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to insert event history: {e}")
            return None

    def log_event_async(self, data: dict):
        """Public method for asynchronous event logging"""
        self._async_write(self.insert_event_history, (data,))

    def log_analytics_async(self, data: dict):
        """Public method for asynchronous analytics logging"""
        self._async_write(self.insert_crowd_analytics, (data,))

    
    def get_recent_crowd_data(self, limit: int = 100):
        """Get recent crowd analytics data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM crowd_analytics 
                    ORDER BY timestamp DESC LIMIT ?
                ''', (limit,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to retrieve crowd data: {e}")
            return []
    
    def get_threat_events(self, limit: int = 50):
        """Get recent threat events"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM threat_events 
                    ORDER BY timestamp DESC LIMIT ?
                ''', (limit,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to retrieve threat events: {e}")
            return []

    def get_event_history(self, limit: int = 500, filters: dict = None):
        """Get event history with optional filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = "SELECT * FROM event_history"
                params = []
                
                if filters:
                    conditions = []
                    if filters.get('threat_level'):
                        conditions.append("threat_level = ?")
                        params.append(filters['threat_level'])
                    if filters.get('camera_source'):
                        conditions.append("camera_source = ?")
                        params.append(filters['camera_source'])
                    if filters.get('event_type'):
                        conditions.append("event_type LIKE ?")
                        params.append(f"%{filters['event_type']}%")
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                        
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                # Fetch with column names
                columns = [column[0] for column in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                return results
        except Exception as e:
            logger.error(f"Failed to retrieve event history: {e}")
            return []

    def insert_camera(self, data: dict):
        """Register a new camera"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO cameras (name, type, ip_address, rtsp_url, location, area)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('name'),
                    data.get('type'),
                    data.get('ip_address'),
                    data.get('rtsp_url'),
                    data.get('location'),
                    data.get('area')
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to insert camera: {e}")
            return None

    def get_cameras(self):
        """Get all registered cameras"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM cameras")
                columns = [column[0] for column in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to retrieve cameras: {e}")
            return []

    def delete_camera(self, camera_id: int):
        """Delete a camera registration"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to delete camera: {e}")
            return False

# Global database instance
db_manager = DatabaseManager()