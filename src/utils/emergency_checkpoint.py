#!/usr/bin/env python
"""
Emergency Checkpoint Module

This module provides robust emergency checkpoint creation and restoration capabilities
to ensure system resilience during critical operations.
"""

import os
import sys
import json
import shutil
import hashlib
import logging
import traceback
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time

# Configure logging
logger = logging.getLogger("emergency_checkpoint")

class EmergencyCheckpoint:
    """Manages emergency checkpoints for application state and model data."""
    
    # Maximum number of emergency checkpoints to keep
    MAX_CHECKPOINTS = 5
    
    # Format for checkpoint directories
    CHECKPOINT_DIR_FORMAT = "emergency_checkpoint_{timestamp}"
    
    # Timestamp format for checkpoint directories
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    
    def __init__(self, base_dir: str, app_state: Optional[Dict[str, Any]] = None):
        """
        Initialize the emergency checkpoint system.
        
        Args:
            base_dir: Base directory for storing checkpoints
            app_state: Optional application state to use for initialization
        """
        self.base_dir = base_dir
        self.checkpoint_dir = os.path.join(base_dir, "emergency_checkpoints")
        self.app_state = app_state
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info(f"Emergency checkpoint system initialized in {self.checkpoint_dir}")
    
    def create_checkpoint(self, app_state: Dict[str, Any], checkpoint_note: str = "", 
                          critical_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create an emergency checkpoint with application state and critical files.
        
        Args:
            app_state: Application state dictionary
            checkpoint_note: Optional note describing the checkpoint
            critical_files: List of critical files to include in the checkpoint
        
        Returns:
            Dictionary with checkpoint details
        """
        result = {
            "success": False,
            "checkpoint_dir": None,
            "error": None,
            "checkpoint_id": None
        }
        
        try:
            # Generate timestamp and checkpoint ID
            timestamp = datetime.now().strftime(self.TIMESTAMP_FORMAT)
            checkpoint_id = f"{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
            
            # Create checkpoint directory
            checkpoint_dir = os.path.join(self.checkpoint_dir, 
                                         self.CHECKPOINT_DIR_FORMAT.format(timestamp=timestamp))
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Add metadata to app_state
            checkpoint_app_state = app_state.copy()
            checkpoint_app_state.update({
                "checkpoint_metadata": {
                    "checkpoint_id": checkpoint_id,
                    "timestamp": timestamp,
                    "note": checkpoint_note,
                    "creation_time": datetime.now().isoformat(),
                    "is_emergency_checkpoint": True
                }
            })
            
            # Save application state
            state_file = os.path.join(checkpoint_dir, "app_state.json")
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_app_state, f, indent=4)
            
            # Calculate and save state checksum
            checksum = hashlib.md5(json.dumps(checkpoint_app_state, sort_keys=True).encode()).hexdigest()
            checksum_file = os.path.join(checkpoint_dir, "checksum.json")
            with open(checksum_file, "w", encoding="utf-8") as f:
                json.dump({
                    "app_state_checksum": checksum,
                    "checkpoint_id": checkpoint_id,
                    "timestamp": timestamp
                }, f, indent=4)
            
            # Copy critical files if provided
            if critical_files:
                files_dir = os.path.join(checkpoint_dir, "files")
                os.makedirs(files_dir, exist_ok=True)
                
                copied_files = []
                for file_path in critical_files:
                    if os.path.exists(file_path):
                        try:
                            dest_path = os.path.join(files_dir, os.path.basename(file_path))
                            shutil.copy2(file_path, dest_path)
                            copied_files.append(os.path.basename(file_path))
                        except Exception as e:
                            logger.warning(f"Failed to copy critical file {file_path}: {e}")
                
                # Save copied files list
                files_list_path = os.path.join(checkpoint_dir, "files_list.json")
                with open(files_list_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "copied_files": copied_files,
                        "original_paths": critical_files
                    }, f, indent=4)
            
            # Create a checkpoint summary file
            summary_file = os.path.join(checkpoint_dir, "checkpoint_summary.json")
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump({
                    "checkpoint_id": checkpoint_id,
                    "timestamp": timestamp,
                    "note": checkpoint_note,
                    "creation_time": datetime.now().isoformat(),
                    "includes_critical_files": bool(critical_files),
                    "num_critical_files": len(critical_files) if critical_files else 0
                }, f, indent=4)
            
            logger.info(f"Created emergency checkpoint: {checkpoint_id} in {checkpoint_dir}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            # Update result
            result["success"] = True
            result["checkpoint_dir"] = checkpoint_dir
            result["checkpoint_id"] = checkpoint_id
            
            return result
        
        except Exception as e:
            error_message = f"Failed to create emergency checkpoint: {str(e)}"
            logger.error(error_message)
            logger.debug(traceback.format_exc())
            
            result["error"] = error_message
            return result
    
    def restore_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Restore application state from an emergency checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to restore
            
        Returns:
            Dictionary with restoration details
        """
        result = {
            "success": False,
            "app_state": None,
            "error": None,
            "restored_files": []
        }
        
        try:
            # Find the checkpoint directory
            checkpoint_dir = self._find_checkpoint_by_id(checkpoint_id)
            if not checkpoint_dir:
                result["error"] = f"Checkpoint with ID {checkpoint_id} not found"
                return result
            
            # Verify checkpoint integrity
            integrity_result = self._verify_checkpoint_integrity(checkpoint_dir)
            if not integrity_result["success"]:
                result["error"] = f"Checkpoint integrity check failed: {integrity_result['error']}"
                return result
            
            # Load application state
            state_file = os.path.join(checkpoint_dir, "app_state.json")
            with open(state_file, "r", encoding="utf-8") as f:
                app_state = json.load(f)
            
            # Check if there are files to restore
            files_list_path = os.path.join(checkpoint_dir, "files_list.json")
            if os.path.exists(files_list_path):
                with open(files_list_path, "r", encoding="utf-8") as f:
                    files_data = json.load(f)
                
                files_dir = os.path.join(checkpoint_dir, "files")
                if os.path.exists(files_dir):
                    # Make backup of current files before restoring
                    for i, original_path in enumerate(files_data.get("original_paths", [])):
                        if os.path.exists(original_path):
                            backup_path = f"{original_path}.bak.{datetime.now().strftime(self.TIMESTAMP_FORMAT)}"
                            try:
                                shutil.copy2(original_path, backup_path)
                                logger.info(f"Created backup of {original_path} at {backup_path}")
                            except Exception as e:
                                logger.warning(f"Failed to backup {original_path}: {e}")
                    
                    # Restore files
                    for i, file_name in enumerate(files_data.get("copied_files", [])):
                        source_path = os.path.join(files_dir, file_name)
                        if os.path.exists(source_path) and i < len(files_data.get("original_paths", [])):
                            target_path = files_data["original_paths"][i]
                            try:
                                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                                shutil.copy2(source_path, target_path)
                                result["restored_files"].append(target_path)
                                logger.info(f"Restored file {target_path}")
                            except Exception as e:
                                logger.warning(f"Failed to restore {target_path}: {e}")
            
            result["success"] = True
            result["app_state"] = app_state
            
            logger.info(f"Successfully restored checkpoint: {checkpoint_id}")
            return result
            
        except Exception as e:
            error_message = f"Failed to restore checkpoint: {str(e)}"
            logger.error(error_message)
            logger.debug(traceback.format_exc())
            
            result["error"] = error_message
            return result
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available emergency checkpoints.
        
        Returns:
            List of dictionaries with checkpoint metadata
        """
        checkpoints = []
        
        try:
            # List all directories in the checkpoint directory
            for entry in os.listdir(self.checkpoint_dir):
                checkpoint_path = os.path.join(self.checkpoint_dir, entry)
                if os.path.isdir(checkpoint_path):
                    # Try to load summary file
                    summary_file = os.path.join(checkpoint_path, "checkpoint_summary.json")
                    if os.path.exists(summary_file):
                        try:
                            with open(summary_file, "r", encoding="utf-8") as f:
                                summary = json.load(f)
                            
                            # Add directory path to summary
                            summary["directory"] = checkpoint_path
                            
                            # Check integrity
                            integrity_result = self._verify_checkpoint_integrity(checkpoint_path)
                            summary["integrity_verified"] = integrity_result["success"]
                            if not integrity_result["success"]:
                                summary["integrity_error"] = integrity_result["error"]
                            
                            checkpoints.append(summary)
                        except Exception as e:
                            logger.warning(f"Failed to load checkpoint summary from {summary_file}: {e}")
        
        except Exception as e:
            logger.error(f"Error listing checkpoints: {e}")
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete an emergency checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find the checkpoint directory
            checkpoint_dir = self._find_checkpoint_by_id(checkpoint_id)
            if not checkpoint_dir:
                logger.warning(f"Checkpoint with ID {checkpoint_id} not found")
                return False
            
            # Delete the checkpoint directory
            shutil.rmtree(checkpoint_dir)
            logger.info(f"Deleted checkpoint: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    def _find_checkpoint_by_id(self, checkpoint_id: str) -> Optional[str]:
        """
        Find a checkpoint directory by ID.
        
        Args:
            checkpoint_id: Checkpoint ID to find
            
        Returns:
            Path to checkpoint directory or None if not found
        """
        try:
            for entry in os.listdir(self.checkpoint_dir):
                checkpoint_path = os.path.join(self.checkpoint_dir, entry)
                if os.path.isdir(checkpoint_path):
                    # Check summary file
                    summary_file = os.path.join(checkpoint_path, "checkpoint_summary.json")
                    if os.path.exists(summary_file):
                        try:
                            with open(summary_file, "r", encoding="utf-8") as f:
                                summary = json.load(f)
                            
                            if summary.get("checkpoint_id") == checkpoint_id:
                                return checkpoint_path
                        except:
                            pass
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding checkpoint {checkpoint_id}: {e}")
            return None
    
    def _verify_checkpoint_integrity(self, checkpoint_dir: str) -> Dict[str, Any]:
        """
        Verify the integrity of a checkpoint.
        
        Args:
            checkpoint_dir: Path to checkpoint directory
            
        Returns:
            Dictionary with integrity check results
        """
        result = {
            "success": False,
            "error": None
        }
        
        try:
            # Check if required files exist
            required_files = ["app_state.json", "checksum.json", "checkpoint_summary.json"]
            for file_name in required_files:
                file_path = os.path.join(checkpoint_dir, file_name)
                if not os.path.exists(file_path):
                    result["error"] = f"Required file missing: {file_name}"
                    return result
            
            # Load app state
            state_file = os.path.join(checkpoint_dir, "app_state.json")
            with open(state_file, "r", encoding="utf-8") as f:
                app_state = json.load(f)
            
            # Load checksum data
            checksum_file = os.path.join(checkpoint_dir, "checksum.json")
            with open(checksum_file, "r", encoding="utf-8") as f:
                checksum_data = json.load(f)
            
            # Verify app_state checksum
            calculated_checksum = hashlib.md5(json.dumps(app_state, sort_keys=True).encode()).hexdigest()
            stored_checksum = checksum_data.get("app_state_checksum")
            
            if calculated_checksum != stored_checksum:
                result["error"] = "App state checksum mismatch"
                return result
            
            # Check files if they exist
            files_list_path = os.path.join(checkpoint_dir, "files_list.json")
            if os.path.exists(files_list_path):
                with open(files_list_path, "r", encoding="utf-8") as f:
                    files_data = json.load(f)
                
                files_dir = os.path.join(checkpoint_dir, "files")
                if os.path.exists(files_dir):
                    for file_name in files_data.get("copied_files", []):
                        file_path = os.path.join(files_dir, file_name)
                        if not os.path.exists(file_path):
                            result["error"] = f"Saved file missing: {file_name}"
                            return result
            
            result["success"] = True
            return result
            
        except Exception as e:
            result["error"] = f"Error verifying checkpoint integrity: {str(e)}"
            return result
    
    def _cleanup_old_checkpoints(self):
        """
        Remove old checkpoints, keeping only the most recent ones.
        """
        try:
            # Get list of checkpoints sorted by timestamp
            checkpoints = self.list_checkpoints()
            
            # If we have more than MAX_CHECKPOINTS, delete the oldest ones
            if len(checkpoints) > self.MAX_CHECKPOINTS:
                # Sort by timestamp (oldest first)
                checkpoints.sort(key=lambda x: x.get("timestamp", ""))
                
                # Delete the oldest checkpoints
                for checkpoint in checkpoints[:-self.MAX_CHECKPOINTS]:
                    checkpoint_id = checkpoint.get("checkpoint_id")
                    if checkpoint_id:
                        self.delete_checkpoint(checkpoint_id)
                        logger.info(f"Deleted old checkpoint: {checkpoint_id}")
        
        except Exception as e:
            logger.error(f"Error cleaning up old checkpoints: {e}")

# Initialize singleton instance
_checkpoint_manager = None

def get_checkpoint_manager(base_dir: str = None, app_state: Dict[str, Any] = None) -> EmergencyCheckpoint:
    """
    Get or initialize the checkpoint manager singleton.
    
    Args:
        base_dir: Base directory for checkpoints (only used on first call)
        app_state: Application state (only used on first call)
        
    Returns:
        EmergencyCheckpoint: The checkpoint manager instance
    """
    global _checkpoint_manager
    
    if _checkpoint_manager is None:
        if base_dir is None:
            # Try to determine a reasonable default
            if hasattr(sys, 'frozen'):
                # Running as executable
                base_dir = os.path.dirname(sys.executable)
            else:
                # Running as script
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        
        _checkpoint_manager = EmergencyCheckpoint(base_dir, app_state)
    
    return _checkpoint_manager

def create_emergency_checkpoint(app_state: Dict[str, Any], checkpoint_note: str = "",
                               critical_files: List[str] = None, base_dir: str = None) -> Dict[str, Any]:
    """
    Create an emergency checkpoint with application state and critical files.
    
    Args:
        app_state: Application state dictionary
        checkpoint_note: Optional note describing the checkpoint
        critical_files: List of critical files to include in the checkpoint
        base_dir: Optional base directory for checkpoints
    
    Returns:
        Dictionary with checkpoint details
    """
    manager = get_checkpoint_manager(base_dir, app_state)
    return manager.create_checkpoint(app_state, checkpoint_note, critical_files)

def restore_emergency_checkpoint(checkpoint_id: str, base_dir: str = None) -> Dict[str, Any]:
    """
    Restore application state from an emergency checkpoint.
    
    Args:
        checkpoint_id: ID of the checkpoint to restore
        base_dir: Optional base directory for checkpoints
    
    Returns:
        Dictionary with restoration details
    """
    manager = get_checkpoint_manager(base_dir)
    return manager.restore_checkpoint(checkpoint_id)

def list_emergency_checkpoints(base_dir: str = None) -> List[Dict[str, Any]]:
    """
    List all available emergency checkpoints.
    
    Args:
        base_dir: Optional base directory for checkpoints
        
    Returns:
        List of dictionaries with checkpoint metadata
    """
    manager = get_checkpoint_manager(base_dir)
    return manager.list_checkpoints()

def delete_emergency_checkpoint(checkpoint_id: str, base_dir: str = None) -> bool:
    """
    Delete an emergency checkpoint.
    
    Args:
        checkpoint_id: ID of the checkpoint to delete
        base_dir: Optional base directory for checkpoints
        
    Returns:
        True if successful, False otherwise
    """
    manager = get_checkpoint_manager(base_dir)
    return manager.delete_checkpoint(checkpoint_id) 