import os
import json
import logging
import datetime
import uuid
import shutil
import zipfile
import platform
from pathlib import Path
import hashlib
import traceback

# Try to import logging utilities
try:
    from src.utils.log_manager import LogManager
    logger = LogManager.get_logger("persistent_logger")
except ImportError:
    # Fallback if imports fail
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("persistent_logger")

class PersistentErrorLogger:
    """
    Error logger that maintains logs across app installations and can
    be collected for later analysis.
    """
    
    def __init__(self):
        # Create a unique installation ID if not already present
        self.installation_id = self._get_or_create_installation_id()
        
        # Set up persistent log directory (outside program directory)
        self.log_dir = self._get_persistent_log_dir()
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Current log file
        self.current_log_file = os.path.join(self.log_dir, "error_log.json")
        
        # Initialize error log if it doesn't exist
        if not os.path.exists(self.current_log_file):
            self._initialize_log_file()
        
        # Standard logger for immediate use
        self.logger = logging.getLogger("persistent_errors")
        self.logger.setLevel(logging.ERROR)
        
        logger.info(f"Persistent logger initialized with ID: {self.installation_id}")
        logger.info(f"Log directory: {self.log_dir}")
    
    def _get_persistent_log_dir(self):
        """Create a path that will survive uninstallation"""
        system = platform.system()
        
        if system == "Windows":
            # Use AppData/Local for Windows
            base_dir = os.path.join(os.environ["LOCALAPPDATA"], "BTC-AI-Logs")
        elif system == "Darwin":  # macOS
            base_dir = os.path.expanduser("~/Library/Application Support/BTC-AI-Logs")
        else:  # Linux and others
            base_dir = os.path.expanduser("~/.btc-ai-logs")
            
        return base_dir
    
    def _get_or_create_installation_id(self):
        """Get or create a unique installation ID"""
        # Get base directory without creating it yet
        system = platform.system()
        if system == "Windows":
            base_dir = os.path.join(os.environ["LOCALAPPDATA"], "BTC-AI-Logs")
        elif system == "Darwin":  # macOS
            base_dir = os.path.expanduser("~/Library/Application Support/BTC-AI-Logs")
        else:  # Linux and others
            base_dir = os.path.expanduser("~/.btc-ai-logs")
            
        id_file = os.path.join(base_dir, "installation_id.txt")
        
        if os.path.exists(id_file):
            with open(id_file, 'r') as f:
                return f.read().strip()
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(id_file), exist_ok=True)
            
            # Generate new ID (combination of machine info and UUID)
            machine_id = platform.node()
            unique_id = str(uuid.uuid4())
            installation_id = hashlib.md5(f"{machine_id}:{unique_id}".encode()).hexdigest()
            
            # Save ID
            with open(id_file, 'w') as f:
                f.write(installation_id)
                
            return installation_id
    
    def _initialize_log_file(self):
        """Initialize a new error log file"""
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "installation_id": self.installation_id,
            "created": datetime.datetime.now().isoformat()
        }
        
        log_data = {
            "system_info": system_info,
            "errors": []
        }
        
        with open(self.current_log_file, 'w') as f:
            json.dump(log_data, f, indent=4)
        
        logger.info(f"Initialized new error log file: {self.current_log_file}")
    
    def log_error(self, error, context="", severity="medium", stack_trace=None, additional_info=None):
        """
        Log an error to the persistent log
        
        Args:
            error: Exception or error string
            context: Where the error occurred
            severity: Error severity (low, medium, high, critical)
            stack_trace: Optional stack trace (if not provided, will be generated)
            additional_info: Any additional context info as a dict
        """
        # Log to standard logger first
        self.logger.error(f"{context}: {error}")
        
        # Ensure severity is a string (handle ErrorSeverity objects)
        if hasattr(severity, 'value'):
            # If it's an enum or has a value attribute
            severity_str = severity.value
        elif hasattr(severity, '__str__'):
            # Convert to string if it's an object
            severity_str = str(severity)
        else:
            # Use as is if it's already a string
            severity_str = severity
        
        # Create error entry with string severity
        error_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": error.__class__.__name__ if isinstance(error, Exception) else "String",
            "message": str(error),
            "context": context,
            "severity": severity_str,
            "stack_trace": stack_trace or self._get_stack_trace(),
            "additional_info": self._ensure_json_serializable(additional_info or {})
        }
        
        # Read current log
        try:
            with open(self.current_log_file, 'r') as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Reinitialize if corrupted or missing
            self._initialize_log_file()
            with open(self.current_log_file, 'r') as f:
                log_data = json.load(f)
        
        # Add error and write back
        log_data["errors"].append(error_entry)
        
        # Check if we need to rotate the log
        if len(log_data["errors"]) > 1000:
            self._rotate_logs()
            # Reinitialize after rotation
            self._initialize_log_file()
            with open(self.current_log_file, 'r') as f:
                log_data = json.load(f)
            log_data["errors"].append(error_entry)
        
        # Write updated log
        with open(self.current_log_file, 'w') as f:
            json.dump(log_data, f, indent=4)
    
    def _ensure_json_serializable(self, obj):
        """
        Ensure an object is JSON serializable by converting non-serializable types to strings
        
        Args:
            obj: Object to make serializable
            
        Returns:
            JSON serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Convert custom objects to dict
            return {
                '__class__': obj.__class__.__name__,
                'attributes': self._ensure_json_serializable(obj.__dict__)
            }
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            # Convert anything else to string
            return str(obj)
    
    def _get_stack_trace(self):
        """Get a formatted stack trace"""
        return traceback.format_exc()
    
    def _rotate_logs(self):
        """Archive the current log file and start fresh"""
        if not os.path.exists(self.current_log_file):
            return
            
        # Create archives directory
        archives_dir = os.path.join(self.log_dir, "archives")
        os.makedirs(archives_dir, exist_ok=True)
        
        # Create timestamped archive name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"error_log_{timestamp}.json"
        archive_path = os.path.join(archives_dir, archive_name)
        
        # Copy current log to archive
        shutil.copy2(self.current_log_file, archive_path)
        
        # Clear current log (will be reinitialized on next write)
        os.remove(self.current_log_file)
        
        logger.info(f"Rotated log file to: {archive_path}")
    
    def export_logs(self, target_path=None):
        """
        Export all logs as a zip file
        
        Args:
            target_path: Where to save the zip file (defaults to user's desktop)
            
        Returns:
            Path to the exported zip file
        """
        # Default to desktop if no path provided
        if not target_path:
            if platform.system() == "Windows":
                desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
            else:
                desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            target_path = os.path.join(desktop, f"btc_ai_logs_{self.installation_id}_{timestamp}.zip")
        
        # Create zip file
        with zipfile.ZipFile(target_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add current log
            if os.path.exists(self.current_log_file):
                zipf.write(self.current_log_file, os.path.basename(self.current_log_file))
            
            # Add archived logs
            archives_dir = os.path.join(self.log_dir, "archives")
            if os.path.exists(archives_dir):
                for file in os.listdir(archives_dir):
                    if file.endswith('.json'):
                        file_path = os.path.join(archives_dir, file)
                        zipf.write(file_path, os.path.join("archives", file))
            
            # Add installation ID
            id_file = os.path.join(self.log_dir, "installation_id.txt")
            if os.path.exists(id_file):
                zipf.write(id_file, os.path.basename(id_file))
        
        logger.info(f"Exported logs to: {target_path}")
        return target_path

# Create a singleton instance
error_logger = PersistentErrorLogger()

def log_persistent_error(error, context="", severity="medium", additional_info=None):
    """Convenience function to log an error"""
    error_logger.log_error(error, context, severity, additional_info=additional_info)

def export_error_logs(target_path=None):
    """Convenience function to export logs"""
    return error_logger.export_logs(target_path)

def get_log_directory_path():
    """Returns the path where logs are stored"""
    return error_logger.log_dir

def open_log_directory():
    """Open the log directory in the system file explorer"""
    log_dir = get_log_directory_path()
    
    try:
        if platform.system() == "Windows":
            os.startfile(log_dir)
        elif platform.system() == "Darwin":  # macOS
            os.system(f'open "{log_dir}"')
        else:  # Linux
            os.system(f'xdg-open "{log_dir}"')
        
        return True
    except Exception as e:
        logger.error(f"Error opening log directory: {e}")
        return False

def create_log_locations_file(output_path=None):
    """
    Create a file with information about log locations
    
    Args:
        output_path: Where to save the file (defaults to desktop)
        
    Returns:
        Path to the created file
    """
    if not output_path:
        if platform.system() == "Windows":
            desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        else:
            desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        output_path = os.path.join(desktop, "btc_ai_log_locations.txt")
    
    try:
        # Get log locations
        error_log_dir = get_log_directory_path()
        
        # Get other log locations
        log_dirs = []
        try:
            from src.utils.log_manager import LogManager
            app_log_dir = LogManager.get_log_directory()
            log_dirs.append(("Application Logs", app_log_dir))
        except ImportError:
            pass
        
        log_dirs.append(("Error Logs", error_log_dir))
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write("BTC AI Log Locations\n")
            f.write("===================\n\n")
            
            for name, path in log_dirs:
                f.write(f"{name}: {path}\n")
                
                # List files in directory
                if os.path.exists(path):
                    f.write("\nFiles:\n")
                    for file in os.listdir(path):
                        file_path = os.path.join(path, file)
                        if os.path.isfile(file_path):
                            size = os.path.getsize(file_path)
                            modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                            f.write(f"- {file} ({size} bytes, {modified})\n")
                    
                    # Check for archives
                    archives_dir = os.path.join(path, "archives")
                    if os.path.exists(archives_dir) and os.path.isdir(archives_dir):
                        f.write("\nArchived Logs:\n")
                        for file in os.listdir(archives_dir):
                            file_path = os.path.join(archives_dir, file)
                            if os.path.isfile(file_path):
                                size = os.path.getsize(file_path)
                                modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                                f.write(f"- archives/{file} ({size} bytes, {modified})\n")
                
                f.write("\n\n")
        
        logger.info(f"Created log locations file at: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error creating log locations file: {e}")
        return None

def merge_log_files(log_files, output_file):
    """
    Merge multiple log files into one
    
    Args:
        log_files: List of log file paths
        output_file: Output file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        merged_data = {
            "merged_info": {
                "created": datetime.datetime.now().isoformat(),
                "source_files": log_files,
                "merged_by": platform.node()
            },
            "errors": []
        }
        
        # Read and merge all files
        for file_path in log_files:
            if os.path.exists(file_path) and file_path.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        log_data = json.load(f)
                        
                    if "errors" in log_data and isinstance(log_data["errors"], list):
                        # Add source file to each error
                        for error in log_data["errors"]:
                            error["source_file"] = os.path.basename(file_path)
                            merged_data["errors"].append(error)
                except Exception as e:
                    logger.error(f"Error reading log file {file_path}: {e}")
        
        # Sort errors by timestamp if present
        merged_data["errors"].sort(
            key=lambda x: x.get("timestamp", "0"),
            reverse=True  # Most recent first
        )
        
        # Write merged data
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=4)
        
        logger.info(f"Merged {len(log_files)} log files into {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error merging log files: {e}")
        return False 