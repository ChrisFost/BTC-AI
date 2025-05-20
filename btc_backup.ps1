#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Automated Rotating Backup System for BTC AI Project
.DESCRIPTION
    Creates time-stamped backups of the BTC AI project and maintains a rotation
    of the 5 most recent backups. Adheres to the BTC AI Development Rules.
.PARAMETER Version
    The version number to include in the backup filename (e.g., "1.4")
.PARAMETER Description
    A brief description of the backup (e.g., "menu_refactor_completed")
.EXAMPLE
    .\btc_backup.ps1 -Version "1.4" -Description "menu_refactor_completed"
    Creates a backup named BTC_AI_YYYY-MM-DD_v1.4_menu_refactor_completed.zip
.NOTES
    This script implements the Automated Rotating Backup requirement from the
    BTC AI Development Ruleset, maintaining exactly 5 backups at all times.
#>

param (
    [Parameter(Mandatory=$true)]
    [string]$Version,
    
    [Parameter(Mandatory=$true)]
    [string]$Description
)

# Configuration
$PROJECT_NAME = "BTC_AI"
$BACKUP_DIR = "C:\Users\chris\OneDrive\Desktop\AI_Version_5_backup"
$SOURCE_DIR = "C:\Users\chris\OneDrive\Desktop\AI_Version_5"
$MAX_BACKUPS = 5

# Ensure the backup directory exists
if (-not (Test-Path $BACKUP_DIR)) {
    Write-Host "Creating backup directory: $BACKUP_DIR"
    New-Item -Path $BACKUP_DIR -ItemType Directory -Force | Out-Null
}

# Generate timestamp and filename
$timestamp = Get-Date -Format "yyyy-MM-dd"
$backupFile = "${PROJECT_NAME}_${timestamp}_v${Version}_${Description}.zip"
$backupPath = Join-Path $BACKUP_DIR $backupFile

# Create the backup
try {
    Write-Host "Creating backup: $backupFile"
    Write-Host "Source: $SOURCE_DIR"
    Write-Host "Destination: $backupPath"
    
    # Compress the directory (PowerShell 5.0+)
    if (Get-Command Compress-Archive -ErrorAction SilentlyContinue) {
        Compress-Archive -Path "$SOURCE_DIR\*" -DestinationPath $backupPath -Force
    } else {
        # Fallback for older PowerShell versions using .NET
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::CreateFromDirectory($SOURCE_DIR, $backupPath)
    }
    
    Write-Host "Backup created successfully!" -ForegroundColor Green
    
    # Check if we have more than MAX_BACKUPS backups
    $allBackups = Get-ChildItem -Path $BACKUP_DIR -Filter "$PROJECT_NAME*.zip" | 
                 Sort-Object -Property LastWriteTime -Descending
    
    # If we have more than MAX_BACKUPS, delete the oldest ones
    if ($allBackups.Count -gt $MAX_BACKUPS) {
        $toDelete = $allBackups | Select-Object -Skip $MAX_BACKUPS
        foreach ($file in $toDelete) {
            Write-Host "Removing old backup: $($file.Name)" -ForegroundColor Yellow
            Remove-Item $file.FullName -Force
        }
    }
    
    # Display current backups
    Write-Host "`nCurrent backups:" -ForegroundColor Cyan
    $currentBackups = Get-ChildItem -Path $BACKUP_DIR -Filter "$PROJECT_NAME*.zip" | 
                     Sort-Object -Property LastWriteTime -Descending |
                     Select-Object -First $MAX_BACKUPS
    
    foreach ($backup in $currentBackups) {
        $size = [math]::Round($backup.Length / 1MB, 2)
        Write-Host "  $($backup.Name) - $($size) MB - $($backup.LastWriteTime)"
    }
    
} catch {
    Write-Host "Error creating backup: $_" -ForegroundColor Red
    exit 1
}

# Return the path to the created backup
return $backupPath 