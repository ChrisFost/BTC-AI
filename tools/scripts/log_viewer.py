#!/usr/bin/env python
"""
BTC-AI Log Viewer and Manager

This utility script helps users view and manage log files created by the BTC-AI application.
It provides capabilities to:
1. View the contents of log files
2. Clear log files
3. Archive old log files
4. Search for specific patterns in log files
"""

import os
import sys
import glob
import time
import re
import zipfile
from datetime import datetime, timedelta
import argparse

# Define paths
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "Logs")
ARCHIVE_DIR = os.path.join(LOG_DIR, "archives")

def ensure_dirs_exist():
    """Ensure log and archive directories exist"""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

def get_log_files():
    """Get a list of all log files"""
    ensure_dirs_exist()
    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    return sorted(log_files)

def list_logs():
    """List all available log files with sizes and timestamps"""
    log_files = get_log_files()
    if not log_files:
        print("No log files found in", LOG_DIR)
        return
    
    print(f"\nLog files in {LOG_DIR}:\n{'-' * 60}")
    print(f"{'ID':<3} {'Size (KB)':<10} {'Last Modified':<20} {'Filename'}")
    print("-" * 60)
    
    for i, log_file in enumerate(log_files):
        filename = os.path.basename(log_file)
        size_kb = os.path.getsize(log_file) / 1024
        mod_time = datetime.fromtimestamp(os.path.getmtime(log_file)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i:<3} {size_kb:<10.2f} {mod_time:<20} {filename}")

def view_log(log_id=None, filename=None, lines=50):
    """View the contents of a log file"""
    log_files = get_log_files()
    
    if not log_files:
        print("No log files found in", LOG_DIR)
        return
    
    # Determine which log file to view
    if filename:
        log_file = os.path.join(LOG_DIR, filename)
        if not os.path.exists(log_file):
            print(f"Error: Log file '{filename}' not found")
            return
    elif log_id is not None:
        if log_id < 0 or log_id >= len(log_files):
            print(f"Error: Invalid log ID. Valid range is 0-{len(log_files)-1}")
            return
        log_file = log_files[log_id]
    else:
        # Default to the most recent log file (UI log)
        ui_logs = [f for f in log_files if "ui.log" in os.path.basename(f)]
        if ui_logs:
            log_file = ui_logs[0]
        else:
            log_file = log_files[-1]  # Use the last log file in the sorted list
    
    # Read the log file
    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.readlines()
        
        # Get only the last N lines
        log_tail = content[-lines:] if lines < len(content) else content
        
        print(f"\nDisplaying last {len(log_tail)} lines of {os.path.basename(log_file)}:")
        print("-" * 80)
        for line in log_tail:
            print(line.rstrip())
        print("-" * 80)
        print(f"Total lines in log: {len(content)}")
    except Exception as e:
        print(f"Error reading log file: {str(e)}")

def clear_log(log_id=None, filename=None):
    """Clear the contents of a log file"""
    log_files = get_log_files()
    
    if not log_files:
        print("No log files found in", LOG_DIR)
        return
    
    # Determine which log file to clear
    if filename:
        log_file = os.path.join(LOG_DIR, filename)
        if not os.path.exists(log_file):
            print(f"Error: Log file '{filename}' not found")
            return
    elif log_id is not None:
        if log_id < 0 or log_id >= len(log_files):
            print(f"Error: Invalid log ID. Valid range is 0-{len(log_files)-1}")
            return
        log_file = log_files[log_id]
    else:
        print("Error: Must specify either log_id or filename to clear")
        return
    
    # Confirm with the user
    filename = os.path.basename(log_file)
    confirm = input(f"Are you sure you want to clear the contents of '{filename}'? (y/n): ")
    
    if confirm.lower() != 'y':
        print("Operation cancelled")
        return
    
    # Clear the log file
    try:
        with open(log_file, 'w') as f:
            f.write("")
        print(f"Successfully cleared {filename}")
    except Exception as e:
        print(f"Error clearing log file: {str(e)}")

def archive_logs(days=30):
    """Archive log files older than specified days"""
    log_files = get_log_files()
    
    if not log_files:
        print("No log files found in", LOG_DIR)
        return
    
    cutoff_date = datetime.now() - timedelta(days=days)
    archive_name = os.path.join(ARCHIVE_DIR, f"logs_before_{cutoff_date.strftime('%Y%m%d')}.zip")
    
    # Find files to archive
    files_to_archive = []
    for log_file in log_files:
        file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
        if file_time < cutoff_date:
            files_to_archive.append(log_file)
    
    if not files_to_archive:
        print(f"No log files older than {days} days found")
        return
    
    # Confirm with the user
    confirm = input(f"Archive {len(files_to_archive)} log files older than {days} days? (y/n): ")
    
    if confirm.lower() != 'y':
        print("Operation cancelled")
        return
    
    # Create the archive
    try:
        with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for log_file in files_to_archive:
                zipf.write(log_file, os.path.basename(log_file))
                
                # Remove the original file after archiving
                os.remove(log_file)
        
        print(f"Successfully archived {len(files_to_archive)} log files to {os.path.basename(archive_name)}")
    except Exception as e:
        print(f"Error archiving log files: {str(e)}")

def search_logs(pattern, case_sensitive=False):
    """Search log files for a specific pattern"""
    log_files = get_log_files()
    
    if not log_files:
        print("No log files found in", LOG_DIR)
        return
    
    # Compile the regex pattern
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error:
        print(f"Error: Invalid regular expression pattern: {pattern}")
        return
    
    print(f"Searching for '{pattern}' in log files...")
    
    # Search each log file
    results = []
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                
            file_results = []
            for i, line in enumerate(lines):
                if regex.search(line):
                    file_results.append((i+1, line.strip()))
            
            if file_results:
                results.append((log_file, file_results))
        except Exception as e:
            print(f"Error searching {os.path.basename(log_file)}: {str(e)}")
    
    # Display results
    if not results:
        print(f"No matches found for '{pattern}'")
        return
    
    total_matches = sum(len(file_results) for _, file_results in results)
    print(f"\nFound {total_matches} matches in {len(results)} files:")
    
    for log_file, file_results in results:
        filename = os.path.basename(log_file)
        print(f"\n{filename} ({len(file_results)} matches):")
        print("-" * 80)
        for line_num, line_text in file_results[:10]:  # Limit to 10 matches per file
            print(f"{line_num:5}: {line_text}")
        
        if len(file_results) > 10:
            print(f"... and {len(file_results) - 10} more matches")

def main():
    """Main function for log viewer and manager"""
    parser = argparse.ArgumentParser(description="BTC-AI Log Viewer and Manager")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all log files')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View a log file')
    view_parser.add_argument('-i', '--id', type=int, help='ID of the log file to view')
    view_parser.add_argument('-f', '--file', help='Name of the log file to view')
    view_parser.add_argument('-n', '--lines', type=int, default=50, help='Number of lines to display (default: 50)')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear a log file')
    clear_parser.add_argument('-i', '--id', type=int, help='ID of the log file to clear')
    clear_parser.add_argument('-f', '--file', help='Name of the log file to clear')
    
    # Archive command
    archive_parser = subparsers.add_parser('archive', help='Archive old log files')
    archive_parser.add_argument('-d', '--days', type=int, default=30, help='Archive logs older than this many days (default: 30)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search log files')
    search_parser.add_argument('pattern', help='Pattern to search for (supports regex)')
    search_parser.add_argument('-c', '--case-sensitive', action='store_true', help='Make the search case-sensitive')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == 'list':
        list_logs()
    elif args.command == 'view':
        view_log(args.id, args.file, args.lines)
    elif args.command == 'clear':
        clear_log(args.id, args.file)
    elif args.command == 'archive':
        archive_logs(args.days)
    elif args.command == 'search':
        search_logs(args.pattern, args.case_sensitive)
    else:
        # Default to showing help if no command is specified
        parser.print_help()

if __name__ == "__main__":
    main() 