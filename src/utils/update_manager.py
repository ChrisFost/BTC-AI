"""
Update Manager for BTC-AI Application

This module provides functionality to check for updates, download them,
and apply them to the BTC-AI application.

Features:
- Version checking against a remote server
- Secure download of update packages
- Verification of package integrity
- Application of updates with backup
- Rollback capability if update fails
"""

import os
import sys
import json
import time
import zipfile
import hashlib
import tempfile
import shutil
import threading
import urllib.request
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from pathlib import Path

# Import logging
try:
    from src.utils.log_manager import LogManager, get_logger
    logger = get_logger('update_manager')
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('update_manager')

# Constants
DEFAULT_UPDATE_URL = "https://btc-ai-updates.example.com/api/updates"
DEFAULT_VERSION_FILE = "version.json"
UPDATE_CHECK_INTERVAL = 3600  # 1 hour in seconds
BACKUP_DIR = "Backups"

class Version:
    """Class representing a version with major.minor.patch format"""
    
    def __init__(self, version_str: str):
        """Initialize version from string in format 'major.minor.patch'"""
        parts = version_str.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}. Expected 'major.minor.patch'")
        
        try:
            self.major = int(parts[0])
            self.minor = int(parts[1])
            self.patch = int(parts[2])
        except ValueError:
            raise ValueError(f"Invalid version component in {version_str}. All components must be integers")
    
    def __str__(self) -> str:
        """Convert version to string"""
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __eq__(self, other) -> bool:
        """Check if versions are equal"""
        if not isinstance(other, Version):
            other = Version(str(other))
        return (self.major == other.major and 
                self.minor == other.minor and 
                self.patch == other.patch)
    
    def __gt__(self, other) -> bool:
        """Check if this version is greater than other"""
        if not isinstance(other, Version):
            other = Version(str(other))
        
        if self.major > other.major:
            return True
        if self.major < other.major:
            return False
        
        # Major versions are equal, check minor
        if self.minor > other.minor:
            return True
        if self.minor < other.minor:
            return False
        
        # Minor versions are equal, check patch
        return self.patch > other.patch
    
    def __lt__(self, other) -> bool:
        """Check if this version is less than other"""
        if not isinstance(other, Version):
            other = Version(str(other))
        
        return not (self > other or self == other)
    
    def __ge__(self, other) -> bool:
        """Check if this version is greater than or equal to other"""
        return self > other or self == other
    
    def __le__(self, other) -> bool:
        """Check if this version is less than or equal to other"""
        return self < other or self == other


class UpdateManager:
    """Manager class for handling application updates"""
    
    def __init__(
        self,
        app_dir: str = None,
        current_version: str = "0.0.1",
        update_url: str = DEFAULT_UPDATE_URL,
        version_file: str = DEFAULT_VERSION_FILE,
        check_interval: int = UPDATE_CHECK_INTERVAL,
        auto_check: bool = True
    ):
        """
        Initialize the update manager
        
        Args:
            app_dir: Application directory (defaults to current directory)
            current_version: Current application version
            update_url: URL to check for updates
            version_file: Path to version file relative to app_dir
            check_interval: Interval between update checks in seconds
            auto_check: Whether to start automatic update checking
        """
        # Set up application directory
        self.app_dir = app_dir or os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        # Version information
        self.current_version = Version(current_version)
        self.latest_version = None
        self.update_info = None
        self.version_file_path = os.path.join(self.app_dir, version_file)
        
        # Update configuration
        self.update_url = update_url
        self.check_interval = check_interval
        
        # State tracking
        self.last_check_time = 0
        self.check_in_progress = False
        self.download_in_progress = False
        self.update_in_progress = False
        self.update_thread = None
        self.background_enabled = auto_check
        
        # Callbacks
        self.on_update_available = None
        self.on_update_progress = None
        self.on_update_complete = None
        self.on_update_error = None
        
        # Create backup directory if it doesn't exist
        self.backup_dir = os.path.join(self.app_dir, BACKUP_DIR)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Load current version from file if available
        self._load_version()
        
        # Start automatic update checking if enabled
        if auto_check:
            self.start_background_check()
    
    def _load_version(self) -> None:
        """Load the current version from version file if it exists"""
        if os.path.exists(self.version_file_path):
            try:
                with open(self.version_file_path, 'r') as f:
                    version_data = json.load(f)
                    if 'version' in version_data:
                        self.current_version = Version(version_data['version'])
                        logger.info(f"Loaded current version: {self.current_version}")
            except Exception as e:
                logger.error(f"Error loading version file: {str(e)}")
        else:
            # Save the initial version file
            self._save_version()
    
    def _save_version(self) -> None:
        """Save the current version to the version file"""
        try:
            version_data = {
                'version': str(self.current_version),
                'last_updated': datetime.now().isoformat(),
                'build_info': {
                    'platform': sys.platform,
                    'python_version': sys.version
                }
            }
            
            with open(self.version_file_path, 'w') as f:
                json.dump(version_data, f, indent=2)
            
            logger.info(f"Saved current version: {self.current_version}")
        except Exception as e:
            logger.error(f"Error saving version file: {str(e)}")
    
    def get_current_version(self) -> str:
        """Get the current application version"""
        return str(self.current_version)
    
    def check_for_updates(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if updates are available
        
        Returns:
            Tuple containing:
                - Boolean indicating if update is available
                - Update information dictionary (or None if no update)
        """
        if self.check_in_progress:
            logger.warning("Update check already in progress")
            return (False, None)
        
        self.check_in_progress = True
        
        try:
            logger.info(f"Checking for updates at {self.update_url}")
            
            # Simulated API call for now
            # In a real app, this would make an HTTPS request to the update server
            # response = requests.get(self.update_url)
            # response_data = response.json()
            
            # For demonstration purposes, simulate a response
            # In production, replace with actual API call
            time.sleep(1)  # Simulate network delay
            response_data = {
                "latest_version": "0.0.2",
                "min_required_version": "0.0.1",
                "release_date": datetime.now().isoformat(),
                "download_url": f"{self.update_url}/downloads/btc-ai-0.0.2.zip",
                "changelog": "- Added update mechanism\n- Fixed bug in logging system",
                "checksum": "0123456789abcdef0123456789abcdef",
                "size_bytes": 1500000,
                "required": False,
                "auto_update": True
            }
            
            # Update the last check time
            self.last_check_time = time.time()
            
            # Parse the version
            try:
                self.latest_version = Version(response_data["latest_version"])
            except ValueError as e:
                logger.error(f"Invalid version format from server: {e}")
                self.check_in_progress = False
                return (False, None)
            
            # Check if update is available
            if self.latest_version > self.current_version:
                logger.info(f"Update available: {self.latest_version} (current: {self.current_version})")
                self.update_info = response_data
                
                # Call callback if registered
                if self.on_update_available and callable(self.on_update_available):
                    self.on_update_available(self.update_info)
                
                self.check_in_progress = False
                return (True, response_data)
            else:
                logger.info(f"No updates available. Current version: {self.current_version}")
                self.check_in_progress = False
                return (False, None)
            
        except Exception as e:
            logger.error(f"Error checking for updates: {str(e)}")
            self.check_in_progress = False
            return (False, None)
    
    def start_background_check(self) -> None:
        """Start background thread for checking updates periodically"""
        if self.update_thread and self.update_thread.is_alive():
            logger.warning("Update thread already running")
            return
        
        self.background_enabled = True
        self.update_thread = threading.Thread(target=self._background_check_worker, daemon=True)
        self.update_thread.start()
        logger.info("Started background update checking")
    
    def stop_background_check(self) -> None:
        """Stop the background update check thread"""
        self.background_enabled = False
        logger.info("Stopped background update checking")
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
    
    def _background_check_worker(self) -> None:
        """Worker function for the background update thread"""
        while self.background_enabled:
            # Check if it's time to check for updates
            if time.time() - self.last_check_time >= self.check_interval:
                try:
                    update_available, _ = self.check_for_updates()
                    # For auto-update functionality:
                    # if update_available and self.update_info.get('auto_update', False):
                    #     self.download_and_apply_update()
                except Exception as e:
                    logger.error(f"Error in background update check: {str(e)}")
            
            # Sleep for a while
            for _ in range(60):  # Check for termination every second
                if not self.background_enabled:
                    break
                time.sleep(1)
    
    def download_update(self, callback: Callable[[int, int], None] = None) -> Optional[str]:
        """
        Download the latest update
        
        Args:
            callback: Function to call with progress updates (bytes_downloaded, total_bytes)
            
        Returns:
            Path to the downloaded update package or None if download failed
        """
        if not self.update_info:
            logger.error("No update information available. Call check_for_updates first.")
            return None
        
        if self.download_in_progress:
            logger.warning("Download already in progress")
            return None
        
        self.download_in_progress = True
        
        try:
            download_url = self.update_info.get("download_url")
            expected_checksum = self.update_info.get("checksum")
            expected_size = self.update_info.get("size_bytes")
            
            if not download_url:
                logger.error("No download URL in update information")
                self.download_in_progress = False
                return None
            
            logger.info(f"Downloading update from {download_url}")
            
            # Create a temporary file for the download
            temp_dir = tempfile.gettempdir()
            download_path = os.path.join(temp_dir, f"btc-ai-update-{self.latest_version}.zip")
            
            # For demonstration purposes, simulate download
            # In production, replace with actual download code
            def simulated_download():
                chunk_size = 8192
                total_size = expected_size or 1000000
                downloaded = 0
                
                with open(download_path, 'wb') as f:
                    while downloaded < total_size:
                        # Simulate download chunk
                        chunk = b'x' * min(chunk_size, total_size - downloaded)
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Report progress
                        if callback:
                            callback(downloaded, total_size)
                        
                        # Simulate network delay
                        time.sleep(0.01)
            
            # Simulate download process
            simulated_download()
            
            # In a real implementation:
            # with urllib.request.urlopen(download_url) as response, open(download_path, 'wb') as out_file:
            #     file_size = int(response.info().get('Content-Length', 0))
            #     downloaded = 0
            #     block_size = 8192
            #     
            #     while True:
            #         buffer = response.read(block_size)
            #         if not buffer:
            #             break
            #             
            #         downloaded += len(buffer)
            #         out_file.write(buffer)
            #         
            #         if callback:
            #             callback(downloaded, file_size)
            
            logger.info(f"Download completed: {download_path}")
            
            # Verify download size
            actual_size = os.path.getsize(download_path)
            if expected_size and actual_size != expected_size:
                logger.error(f"Size mismatch: expected {expected_size}, got {actual_size}")
                self.download_in_progress = False
                return None
            
            # Verify checksum (simulated)
            if expected_checksum:
                # Simulate checksum verification
                # In production, calculate actual checksum
                simulated_checksum = "0123456789abcdef0123456789abcdef"
                
                # If checksums don't match
                if simulated_checksum != expected_checksum:
                    logger.error(f"Checksum mismatch: expected {expected_checksum}, got {simulated_checksum}")
                    self.download_in_progress = False
                    return None
            
            self.download_in_progress = False
            return download_path
            
        except Exception as e:
            logger.error(f"Error downloading update: {str(e)}")
            self.download_in_progress = False
            return None
    
    def apply_update(self, update_path: str) -> bool:
        """
        Apply downloaded update
        
        Args:
            update_path: Path to the downloaded update package
            
        Returns:
            True if update was successfully applied, False otherwise
        """
        if self.update_in_progress:
            logger.warning("Update already in progress")
            return False
        
        if not update_path or not os.path.exists(update_path):
            logger.error(f"Update package not found: {update_path}")
            return False
        
        self.update_in_progress = True
        
        try:
            # Create a timestamped backup directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}")
            os.makedirs(backup_path, exist_ok=True)
            
            logger.info(f"Creating backup at {backup_path}")
            
            # Create a backup of critical files
            # In a real implementation, this would be more sophisticated
            # to only backup files that will be changed
            critical_dirs = ["configs", "src"]
            for dir_name in critical_dirs:
                src_dir = os.path.join(self.app_dir, dir_name)
                if os.path.exists(src_dir):
                    dst_dir = os.path.join(backup_path, dir_name)
                    shutil.copytree(src_dir, dst_dir)
            
            # Also backup version file
            if os.path.exists(self.version_file_path):
                shutil.copy2(self.version_file_path, backup_path)
            
            # Extract update package
            logger.info(f"Extracting update from {update_path}")
            
            # Create a temporary directory for extraction
            extract_dir = tempfile.mkdtemp()
            
            with zipfile.ZipFile(update_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Check for an install script in the package
            install_script = os.path.join(extract_dir, "install.py")
            if os.path.exists(install_script):
                logger.info("Running install script")
                # This is potentially dangerous and should be handled carefully
                # Consider using a subprocess with restricted permissions
                # For demonstration, we just simulate running it
                time.sleep(2)  # Simulate script execution
            else:
                # No install script, just copy files
                logger.info("Copying update files")
                
                # Get list of files to exclude from update
                exclude_file = os.path.join(extract_dir, "exclude.txt")
                excludes = []
                if os.path.exists(exclude_file):
                    with open(exclude_file, 'r') as f:
                        excludes = [line.strip() for line in f if line.strip()]
                
                # Copy files from extract directory to app directory
                for root, dirs, files in os.walk(extract_dir):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if d not in excludes and not d.startswith('.')]
                    
                    for file in files:
                        # Skip special files
                        if file in ["install.py", "exclude.txt"] or file.startswith('.'):
                            continue
                        
                        # Get relative path
                        rel_path = os.path.relpath(os.path.join(root, file), extract_dir)
                        if rel_path in excludes:
                            continue
                        
                        # Calculate destination path
                        dst_path = os.path.join(self.app_dir, rel_path)
                        
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        
                        # Copy file
                        shutil.copy2(os.path.join(root, file), dst_path)
            
            # Clean up temporary directory
            shutil.rmtree(extract_dir)
            
            # Update version information
            self.current_version = self.latest_version
            self._save_version()
            
            logger.info(f"Update successfully applied. New version: {self.current_version}")
            
            # Call callback if registered
            if self.on_update_complete and callable(self.on_update_complete):
                self.on_update_complete(str(self.current_version))
            
            self.update_in_progress = False
            return True
            
        except Exception as e:
            logger.error(f"Error applying update: {str(e)}")
            self.update_in_progress = False
            
            # Call error callback if registered
            if self.on_update_error and callable(self.on_update_error):
                self.on_update_error(str(e))
            
            return False
    
    def rollback_update(self, backup_id: str = None) -> bool:
        """
        Rollback to a previous backup
        
        Args:
            backup_id: ID of the backup to rollback to (if None, uses the most recent)
            
        Returns:
            True if rollback was successful, False otherwise
        """
        if self.update_in_progress:
            logger.warning("Update in progress, cannot rollback")
            return False
        
        try:
            # Find available backups
            backups = []
            if os.path.exists(self.backup_dir):
                backups = [d for d in os.listdir(self.backup_dir) 
                          if os.path.isdir(os.path.join(self.backup_dir, d)) 
                          and d.startswith("backup_")]
            
            if not backups:
                logger.error("No backups found")
                return False
            
            # Sort backups by timestamp (latest first)
            backups.sort(reverse=True)
            
            # Select the backup to use
            if backup_id:
                if backup_id not in backups:
                    logger.error(f"Backup {backup_id} not found")
                    return False
                selected_backup = backup_id
            else:
                selected_backup = backups[0]
            
            backup_path = os.path.join(self.backup_dir, selected_backup)
            logger.info(f"Rolling back to backup {selected_backup}")
            
            # Copy files from backup to app directory
            for root, dirs, files in os.walk(backup_path):
                for file in files:
                    # Get relative path
                    rel_path = os.path.relpath(os.path.join(root, file), backup_path)
                    
                    # Calculate destination path
                    dst_path = os.path.join(self.app_dir, rel_path)
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(os.path.join(root, file), dst_path)
            
            # Reload version information
            self._load_version()
            
            logger.info(f"Rollback successful. Current version: {self.current_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back update: {str(e)}")
            return False
    
    def download_and_apply_update(self, progress_callback: Callable[[int, int], None] = None) -> bool:
        """
        Download and apply the latest update in one operation
        
        Args:
            progress_callback: Function to call with progress updates
            
        Returns:
            True if update was successfully applied, False otherwise
        """
        # Check for updates if we haven't already
        if not self.update_info:
            update_available, _ = self.check_for_updates()
            if not update_available:
                logger.info("No updates available")
                return False
        
        # Download the update
        update_path = self.download_update(progress_callback)
        if not update_path:
            logger.error("Failed to download update")
            return False
        
        # Apply the update
        return self.apply_update(update_path)


# Create a global instance for convenience
update_manager = None

def initialize(app_dir=None, version="0.0.1", auto_check=True):
    """Initialize the global update manager"""
    global update_manager
    update_manager = UpdateManager(
        app_dir=app_dir,
        current_version=version,
        auto_check=auto_check
    )
    return update_manager

def get_manager():
    """Get the global update manager instance"""
    global update_manager
    if update_manager is None:
        update_manager = initialize()
    return update_manager 