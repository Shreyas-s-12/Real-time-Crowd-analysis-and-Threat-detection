"""
Network Validator Module for Real-Time Crowd Analysis and Threat Detection
"""

import socket
import subprocess
import platform
import psutil # Keep psutil import
import time # Keep time import
from typing import Dict, Optional, List # Keep typing imports
from real_time_crowd_analysis.utils.config import config # Absolute import
from real_time_crowd_analysis.utils.logger import setup_logger # Absolute import

logger = setup_logger("network_validator")

class NetworkValidator:
    """Validates network connections for camera streams"""
    
    def __init__(self):
        self.system_name = platform.system()
        self.last_network_info = {}
    
    def get_system_ip(self) -> str:
        """Get the system's IP address"""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
            return ip
        except Exception:
            return "127.0.0.1"
    
    def get_network_name(self) -> str:
        """Get the current network name (SSID on Windows)"""
        try:
            if self.system_name == "Windows":
                # Get Wi-Fi network name on Windows
                result = subprocess.run(
                    ["netsh", "wlan", "show", "interfaces"], 
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.split('\n'):
                    if "SSID" in line and ":" in line:
                        return line.split(":")[1].strip()
            elif self.system_name == "Linux":
                # Get network name on Linux
                result = subprocess.run(
                    ["iwgetid", "-r"], 
                    capture_output=True, text=True, timeout=5
                )
                return result.stdout.strip() if result.stdout else "Unknown"
            elif self.system_name == "Darwin":  # macOS
                result = subprocess.run(
                    ["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-I"],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.split('\n'):
                    if " SSID" in line:
                        return line.split(":")[1].strip()
            return "Unknown Network"
        except Exception as e:
            logger.debug(f"Could not get network name: {e}")
            return "Unknown"
    
    def is_valid_ip(self, ip: str) -> bool:
        """Validate IP address format"""
        try:
            socket.inet_aton(ip)
            return ip != '127.0.0.1' and not ip.startswith('169.254')
        except socket.error:
            return False
    
    def ping_host(self, host: str, timeout: int = 3) -> bool:
        """Ping a host to check connectivity"""
        try:
            # Determine ping command based on OS
            param = '-n' if self.system_name == 'Windows' else '-c'
            command = ['ping', param, '1', '-w', str(timeout * 1000), host]
            
            result = subprocess.run(
                command, 
                capture_output=True, 
                timeout=timeout + 1
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_host_ip(self, hostname: str) -> str:
        """Get IP address from hostname"""
        try:
            return socket.gethostbyname(hostname)
        except socket.gaierror:
            return ""
    
    def validate_camera_network(self, camera_ip: str, camera_type: str = "unknown") -> Dict[str, any]:
        """
        Validate that camera and system are on the same network
        
        Returns:
            dict with validation results
        """
        system_ip = self.get_system_ip()
        network_name = self.get_network_name()
        
        result = {
            'system_ip': system_ip,
            'camera_ip': camera_ip,
            'network_name': network_name,
            'same_network': False,
            'camera_reachable': False,
            'connection_status': 'UNKNOWN',
            'timestamp': time.time()
        }
        
        # Validate IP format
        if not self.is_valid_ip(camera_ip):
            result['connection_status'] = 'INVALID_IP'
            return result
        
        # Check if on same subnet (simple check for same first 3 octets)
        try:
            system_parts = system_ip.split('.')
            camera_parts = camera_ip.split('.')
            
            if len(system_parts) == 4 and len(camera_parts) == 4:
                same_subnet = (
                    system_parts[0] == camera_parts[0] and
                    system_parts[1] == camera_parts[1] and
                    system_parts[2] == camera_parts[2]
                )
                result['same_network'] = same_subnet
            else:
                result['same_network'] = False
        except Exception:
            result['same_network'] = False
        
        # Check if camera is reachable
        result['camera_reachable'] = self.ping_host(camera_ip, config.NETWORK_TIMEOUT)
        
        # Determine overall connection status
        if result['same_network'] and result['camera_reachable']:
            result['connection_status'] = 'CONNECTED'
        elif not result['same_network'] and result['camera_reachable']:
            result['connection_status'] = 'DIFFERENT_NETWORK'
        elif result['same_network'] and not result['camera_reachable']:
            result['connection_status'] = 'UNREACHABLE'
        else:
            result['connection_status'] = 'FAILED'
        
        return result
    
    def get_connection_strength(self, camera_ip: str) -> float:
        """Estimate connection strength based on ping response time"""
        try:
            import time
            start = time.time()
            reachable = self.ping_host(camera_ip, 2)
            end = time.time()
            
            if reachable:
                latency = (end - start) * 1000  # Convert to milliseconds
                # Convert latency to strength percentage (lower latency = higher strength)
                # Assuming good connection < 50ms, poor > 200ms
                strength = max(0, min(100, 100 - (latency - 50) * 100 / 150))
                return strength
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def monitor_network_change(self, callback=None):
        """Monitor for network changes and call callback if network changes"""
        current_network = self.get_network_name()
        current_ip = self.get_system_ip()
        
        # Check if network has changed
        if (self.last_network_info.get('name') != current_network or 
            self.last_network_info.get('ip') != current_ip):
            
            self.last_network_info = {
                'name': current_network,
                'ip': current_ip,
                'timestamp': time.time()
            }
            
            if callback:
                callback(current_network, current_ip)
            
            logger.info(f"Network changed to: {current_network} ({current_ip})")
            return True
        
        return False
    
    def get_active_network_interfaces(self) -> List[Dict[str, str]]:
        """Get list of active network interfaces"""
        interfaces = []
        try:
            if self.system_name == "Windows":
                result = subprocess.run(
                    ["ipconfig"], 
                    capture_output=True, text=True, timeout=5
                )
                # Parse Windows ipconfig output
                current_adapter = {}
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line and not line.startswith(' '):
                        if 'adapter' in line.lower():
                            if current_adapter:
                                interfaces.append(current_adapter)
                            current_adapter = {'name': line.rstrip(':')}
                    elif 'IPv4 Address' in line:
                        ip = line.split(':')[1].strip().split('(')[0].strip()
                        if ip and ip != '127.0.0.1':
                            current_adapter['ip'] = ip
                    elif 'Default Gateway' in line:
                        gateway = line.split(':')[1].strip()
                        if gateway:
                            current_adapter['gateway'] = gateway
                if current_adapter:
                    interfaces.append(current_adapter)
            else:
                # For Linux/macOS, use ifconfig or ip
                try:
                    result = subprocess.run(
                        ["ip", "addr", "show"], 
                        capture_output=True, text=True, timeout=5
                    )
                    # Simple parsing - in production would be more robust
                    for line in result.stdout.split('\n'):
                        if 'inet ' in line and '127.0.0.1' not in line:
                            parts = line.strip().split()
                            for part in parts:
                                if '/' in part and part.replace('.', '').replace('/', '').isdigit():
                                    ip = part.split('/')[0]
                                    interfaces.append({'ip': ip, 'type': 'inet'})
                except:
                    pass
        except Exception as e:
            logger.debug(f"Could not get network interfaces: {e}")
        
        return interfaces

# Global network validator instance
network_validator = NetworkValidator()

# Convenience functions
def validate_camera_connection(camera_ip: str, camera_type: str = "unknown") -> Dict[str, any]:
    """Validate camera network connection"""
    return network_validator.validate_camera_network(camera_ip, camera_type)

def validate_network(camera_ip: str) -> Dict[str, any]:
    """
    Validate network connectivity for a camera
    
    Args:
        camera_ip: IP address of the camera to validate
        
    Returns:
        Dictionary with validation results including status, network info, etc.
    """
    try:
        print("Checking camera network...")
        result = network_validator.validate_camera_network(camera_ip)
        
        # Add additional validation info
        if result['connection_status'] == 'CONNECTED':
            print("Camera and System are on the Same Network")
            result['validation_message'] = "Camera and System are on the Same Network"
        elif result['connection_status'] == 'DIFFERENT_NETWORK':
            print("Camera is reachable but on different network")
            result['validation_message'] = "Camera is reachable but on different network"
        elif result['connection_status'] == 'UNREACHABLE':
            print("Camera is on same network but unreachable")
            result['validation_message'] = "Camera is on same network but unreachable"
        elif result['connection_status'] == 'FAILED':
            print("Connection Validation Failed")
            result['validation_message'] = "Connection Validation Failed"
        elif result['connection_status'] == 'INVALID_IP':
            print("Invalid IP address")
            result['validation_message'] = "Invalid IP address"
        else:
            print("Wireless Camera Not Available")
            result['validation_message'] = "Wireless Camera Not Available"
            
        return result
    except Exception as e:
        logger.error(f"Error validating network: {e}")
        print(f"Error validating network: {e}")
        return {
            'system_ip': "0.0.0.0",
            'camera_ip': camera_ip,
            'network_name': "Unknown",
            'same_network': False,
            'camera_reachable': False,
            'connection_status': 'ERROR',
            'validation_message': f"Validation error: {str(e)}",
            'timestamp': time.time()
        }

def get_system_network_info() -> Dict[str, any]:
    """Get current system network information"""
    return {
        'ip': network_validator.get_system_ip(),
        'network_name': network_validator.get_network_name(),
        'timestamp': time.time()
    }

def ping_camera(camera_ip: str, timeout: int = 3) -> bool:
    """Ping camera to check connectivity"""
    return network_validator.ping_host(camera_ip, timeout)