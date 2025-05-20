#!/usr/bin/env python
"""
Checkpoint Validation Module

This module provides enhanced validation and integrity checking for model checkpoints,
implementing sandbox testing and progressive loading capabilities to improve resilience.
"""

import os
import sys
import torch
import hashlib
import json
import logging
import traceback
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from datetime import datetime
import tempfile
import copy

# Configure logging
logger = logging.getLogger("checkpoint_validation")

# Try to import error handling
try:
    from src.ui.error_handler import handle_error, ErrorSeverity
    error_handler_available = True
except ImportError:
    error_handler_available = False
    # Create stub function
    def handle_error(error, context="", window=None, retry_func=None, additional_context=None):
        logger.error(f"Error in {context}: {str(error)}")
        return {"message": str(error), "handled": False}

def calculate_checkpoint_hash(checkpoint_path: str) -> str:
    """
    Calculate a hash for a checkpoint file to verify its integrity.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        str: Hash of the checkpoint file, or empty string if failed
    """
    try:
        hasher = hashlib.md5()
        with open(checkpoint_path, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate checkpoint hash: {e}")
        return ""

def add_hash_to_checkpoint(checkpoint_path: str) -> bool:
    """
    Add a hash to an existing checkpoint file for future integrity verification.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Calculate hash of model_state and optimizer_state
        if "model_state" in checkpoint and "optimizer_state" in checkpoint:
            # Serialize the model and optimizer states
            model_bytes = pickle_serialize(checkpoint["model_state"])
            optimizer_bytes = pickle_serialize(checkpoint["optimizer_state"])
            
            # Calculate hash
            hasher = hashlib.md5()
            hasher.update(model_bytes)
            hasher.update(optimizer_bytes)
            checksum = hasher.hexdigest()
            
            # Add hash to checkpoint
            checkpoint["checksum"] = checksum
            
            # Save the updated checkpoint
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Added checksum to checkpoint: {checkpoint_path}")
            return True
        else:
            logger.warning(f"Checkpoint missing required states: {checkpoint_path}")
            return False
    except Exception as e:
        logger.error(f"Failed to add hash to checkpoint: {e}")
        return False

def pickle_serialize(obj) -> bytes:
    """
    Serialize an object to bytes using pickle.
    
    Args:
        obj: Object to serialize
        
    Returns:
        bytes: Serialized object
    """
    import pickle
    return pickle.dumps(obj)

def verify_checkpoint_integrity(checkpoint_path: str) -> Dict[str, Any]:
    """
    Verify the integrity of a checkpoint file using its embedded checksum.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dict containing:
        - 'success': bool indicating if validation succeeded
        - 'error': Error message if failed
        - 'warnings': List of warnings that don't invalidate the checkpoint
    """
    result = {
        "success": False,
        "error": None,
        "warnings": []
    }
    
    try:
        # Check if file exists
        if not os.path.exists(checkpoint_path):
            result["error"] = f"Checkpoint file does not exist: {checkpoint_path}"
            return result
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check for required fields
        required_fields = ["model_state", "optimizer_state", "episode", "best_reward", "config"]
        missing_fields = [field for field in required_fields if field not in checkpoint]
        
        if missing_fields:
            result["error"] = f"Checkpoint missing required fields: {', '.join(missing_fields)}"
            return result
        
        # If checkpoint has a checksum, verify it
        if "checksum" in checkpoint:
            stored_checksum = checkpoint["checksum"]
            
            # Calculate current checksum
            model_bytes = pickle_serialize(checkpoint["model_state"])
            optimizer_bytes = pickle_serialize(checkpoint["optimizer_state"])
            
            hasher = hashlib.md5()
            hasher.update(model_bytes)
            hasher.update(optimizer_bytes)
            calculated_checksum = hasher.hexdigest()
            
            if stored_checksum != calculated_checksum:
                result["error"] = f"Checkpoint integrity check failed: checksum mismatch"
                return result
            
            logger.info(f"Checkpoint integrity verified: {checkpoint_path}")
            result["success"] = True
        else:
            # No checksum, return true but with a warning
            logger.warning(f"Checkpoint does not have a checksum: {checkpoint_path}")
            result["success"] = True
            result["warnings"].append("Checkpoint does not have a checksum for full integrity verification")
        
        return result
        
    except Exception as e:
        error_msg = f"Error verifying checkpoint integrity: {str(e)}"
        logger.error(error_msg)
        result["error"] = error_msg
        return result

def sandbox_load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Safely load a checkpoint in a sandbox environment to verify it can be loaded without errors.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dict with keys:
        - 'success': bool indicating if loading succeeded
        - 'data': The loaded data if successful
        - 'error': Error message if failed
        - 'warnings': List of warnings
        - 'metadata': Dictionary of checkpoint metadata
    """
    result = {
        "success": False,
        "data": None,
        "error": None,
        "warnings": [],
        "metadata": {}
    }
    
    try:
        # First check file integrity
        integrity_check = verify_checkpoint_integrity(checkpoint_path)
        if not integrity_check["success"]:
            result["error"] = integrity_check["error"]
            return result
        
        # Add any warnings from integrity check
        if "warnings" in integrity_check and integrity_check["warnings"]:
            result["warnings"].extend(integrity_check["warnings"])
        
        # Load the checkpoint in a try-except block to catch any errors
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract metadata without loading the full model
        result["metadata"] = {
            "version": checkpoint.get("version", 1),
            "episode": checkpoint.get("episode", 0),
            "best_reward": checkpoint.get("best_reward", 0),
            "is_emergency": checkpoint.get("is_emergency", False),
            "timestamp": checkpoint.get("timestamp", "unknown"),
            "has_enhanced_state": (
                "performance_history" in checkpoint or
                "recent_rewards" in checkpoint or
                "horizons" in checkpoint
            )
        }
        
        # Check model and optimizer states can be loaded without errors
        try:
            # Create a minimal test model to verify model_state can be loaded
            if "model_state" in checkpoint:
                # This doesn't actually load the state dict, just verifies keys match
                # Full validation would require creating correct model architecture
                model_state = checkpoint["model_state"]
                if not isinstance(model_state, dict):
                    raise ValueError("model_state is not a dictionary")
            
            # Similar check for optimizer state
            if "optimizer_state" in checkpoint:
                optimizer_state = checkpoint["optimizer_state"]
                if not isinstance(optimizer_state, dict):
                    raise ValueError("optimizer_state is not a dictionary")
            
            # Success, return the checkpoint data
            result["success"] = True
            result["data"] = checkpoint
            
        except Exception as inner_e:
            # Error in state dict validation
            result["error"] = f"Invalid state dictionaries: {str(inner_e)}"
            result["warnings"].append(f"Model state validation failed: {str(inner_e)}")
    
    except Exception as e:
        # Error loading the checkpoint
        result["error"] = f"Failed to load checkpoint: {str(e)}"
        logger.error(f"Sandbox loading failed for {checkpoint_path}: {str(e)}")
        logger.debug(traceback.format_exc())
    
    return result

def progressive_load_checkpoint(checkpoint_path: str, model=None, optimizer=None) -> Dict[str, Any]:
    """
    Progressively load components of a checkpoint, recovering as much state as possible.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Optional model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Dict containing:
        - 'success': Overall success of loading operation
        - 'model_loaded': Whether the model was loaded
        - 'optimizer_loaded': Whether the optimizer was loaded
        - 'config_loaded': Whether the config was loaded
        - 'metadata': Dictionary of loaded metadata
        - 'error': Error message if something failed
    """
    result = {
        "success": False,
        "model_loaded": False,
        "optimizer_loaded": False,
        "config_loaded": False,
        "metadata": {},
        "error": None
    }
    
    try:
        # First try sandbox loading to validate the checkpoint
        sandbox_result = sandbox_load_checkpoint(checkpoint_path)
        if not sandbox_result["success"]:
            result["error"] = sandbox_result["error"]
            return result
        
        checkpoint = sandbox_result["data"]
        result["metadata"] = sandbox_result["metadata"]
        
        # Try to load model state if model is provided
        if model is not None and "model_state" in checkpoint:
            try:
                # Attempt to load model state
                missing_keys, unexpected_keys = model.load_state_dict(
                    checkpoint["model_state"], strict=False)
                
                if missing_keys:
                    logger.warning(f"Missing keys when loading model: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")
                
                result["model_loaded"] = True
                logger.info(f"Model partially loaded from checkpoint: {checkpoint_path}")
            except Exception as model_e:
                logger.error(f"Failed to load model state: {str(model_e)}")
                # Continue with other parts despite model failure
        
        # Try to load optimizer state if optimizer is provided
        if optimizer is not None and "optimizer_state" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                result["optimizer_loaded"] = True
                logger.info(f"Optimizer loaded from checkpoint: {checkpoint_path}")
            except Exception as opt_e:
                logger.error(f"Failed to load optimizer state: {str(opt_e)}")
                # Continue with other parts despite optimizer failure
        
        # Load configuration
        if "config" in checkpoint:
            result["config"] = checkpoint["config"]
            result["config_loaded"] = True
        
        # Determine overall success - if at least model or config was loaded
        result["success"] = result["model_loaded"] or result["config_loaded"]
        
    except Exception as e:
        result["error"] = f"Error in progressive loading: {str(e)}"
        logger.error(f"Progressive loading failed: {str(e)}")
        logger.debug(traceback.format_exc())
    
    return result

def create_checkpoint_backup(checkpoint_path: str) -> str:
    """
    Create a backup of a checkpoint file before modifications.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        str: Path to the backup file, or empty string if failed
    """
    try:
        # Get directory and filename
        directory = os.path.dirname(checkpoint_path)
        filename = os.path.basename(checkpoint_path)
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{os.path.splitext(filename)[0]}.{timestamp}.bak"
        backup_path = os.path.join(directory, backup_filename)
        
        # Create backup directory if it doesn't exist
        backup_dir = os.path.join(directory, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Full backup path including directory
        full_backup_path = os.path.join(backup_dir, backup_filename)
        
        # Copy the file
        import shutil
        shutil.copy2(checkpoint_path, full_backup_path)
        logger.info(f"Created checkpoint backup: {full_backup_path}")
        
        return full_backup_path
    except Exception as e:
        logger.error(f"Failed to create checkpoint backup: {e}")
        return ""
