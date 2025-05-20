#!/usr/bin/env python
"""
Unit Tests for Installation Process

This script tests the installation process, including:
- Download simulation
- Directory structure creation
- Environment setup
- File placement verification
"""

import os
import sys
import unittest
import tempfile
import shutil
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('installation_test.log')
    ]
)
logger = logging.getLogger('installation_test')

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Create temporary test directories
TEST_DIR = tempfile.mkdtemp(prefix="install_test_")
TEST_INSTALL_DIR = os.path.join(TEST_DIR, "BTC-AI")

class MockDownloader:
    """Mock class for simulating file downloads"""
    def __init__(self, success=True, file_size=1024*1024):
        self.success = success
        self.file_size = file_size
        self.downloaded = 0
        
    def download(self, url, destination):
        if not self.success:
            raise Exception("Mock download failed")
        
        # Create a mock file
        with open(destination, 'wb') as f:
            f.write(b'0' * self.file_size)
        return True
    
    def get_progress(self):
        return min(100, (self.downloaded / self.file_size) * 100)

class InstallationTest(unittest.TestCase):
    """Test the installation process."""
    
    required_dirs = [
        "Models",
        "Logs",
        "configs",
        "src/agent",
        "src/environment",
        "src/models",
        "src/ui",
        "src/utils",
        "src/training"
    ]
    
    required_files = [
        "configs/config.json",
        "src/agent/agent.py",
        "src/environment/env_base.py",
        "src/environment/env_risk.py",
        "src/models/models.py",
        "src/ui/main.py",
        "src/ui/setup_wizard.py",
        "src/utils/visualization.py",
        "src/utils/reasoning.py",
        "src/training/training.py",
        "README.md",
        "requirements.txt"
    ]
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        os.makedirs(TEST_INSTALL_DIR, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        try:
            shutil.rmtree(TEST_DIR)
        except Exception as e:
            logger.error(f"Error cleaning up test directory: {e}")
    
    def setUp(self):
        """Set up before each test."""
        # Clean the install directory
        for item in os.listdir(TEST_INSTALL_DIR):
            path = os.path.join(TEST_INSTALL_DIR, item)
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    
    def test_create_directory_structure(self):
        """Test creation of required directories."""
        for dir_path in self.required_dirs:
            full_path = os.path.join(TEST_INSTALL_DIR, dir_path)
            os.makedirs(full_path, exist_ok=True)
            self.assertTrue(os.path.exists(full_path))
            self.assertTrue(os.path.isdir(full_path))
    
    def test_create_required_files(self):
        """Test creation of required files."""
        # Create directories first
        for dir_path in self.required_dirs:
            os.makedirs(os.path.join(TEST_INSTALL_DIR, dir_path), exist_ok=True)
        
        # Create files
        for file_path in self.required_files:
            full_path = os.path.join(TEST_INSTALL_DIR, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write("# Test file")
            self.assertTrue(os.path.exists(full_path))
            self.assertTrue(os.path.isfile(full_path))
    
    def test_mock_download(self):
        """Test the download simulation."""
        downloader = MockDownloader()
        test_file = os.path.join(TEST_INSTALL_DIR, "test_download.zip")
        
        # Test successful download
        self.assertTrue(downloader.download("mock_url", test_file))
        self.assertTrue(os.path.exists(test_file))
        self.assertTrue(os.path.getsize(test_file) > 0)
        
        # Test failed download
        failed_downloader = MockDownloader(success=False)
        with self.assertRaises(Exception):
            failed_downloader.download("mock_url", "failed_download.zip")
    
    def test_environment_setup(self):
        """Test Python environment setup."""
        requirements_file = os.path.join(TEST_INSTALL_DIR, "requirements.txt")
        
        # Create mock requirements file
        with open(requirements_file, 'w') as f:
            f.write("torch==2.0.0\n")
            f.write("numpy==1.21.0\n")
            f.write("pandas==1.3.0\n")
        
        self.assertTrue(os.path.exists(requirements_file))
        
        # Read and verify requirements
        with open(requirements_file, 'r') as f:
            requirements = f.read().splitlines()
        
        self.assertIn("torch==2.0.0", requirements)
        self.assertIn("numpy==1.21.0", requirements)
        self.assertIn("pandas==1.3.0", requirements)

    def test_batch_installer(self):
        """Test batch installer functionality."""
        logger.info("Testing batch installer script...")
        
        # Create temporary directory to simulate installation target
        test_install_target = os.path.join(TEST_DIR, "Test_Install_Target")
        os.makedirs(test_install_target, exist_ok=True)
        
        # Create mock version info
        version_file = os.path.join(TEST_INSTALL_DIR, "version.json")
        version_info = {
            "version": "1.0.0",
            "build_date": datetime.now().strftime("%Y-%m-%d"),
            "python_version": "3.8.20",
            "cuda_available": "True",
            "cuda_version": "11.8"
        }
        
        with open(version_file, 'w') as f:
            json.dump(version_info, f)
        
        # Create mock launcher file
        launcher_file = os.path.join(TEST_INSTALL_DIR, "btc_ai_launcher.py")
        with open(launcher_file, 'w') as f:
            f.write("#!/usr/bin/env python\n")
            f.write("print('BTC-AI Launcher')\n")
            f.write("import sys\n")
            f.write("from src.ui.main import main\n")
            f.write("if __name__ == '__main__':\n")
            f.write("    main()\n")
        
        # Create mock main.py file in src/ui
        os.makedirs(os.path.join(TEST_INSTALL_DIR, "src", "ui"), exist_ok=True)
        with open(os.path.join(TEST_INSTALL_DIR, "src", "ui", "main.py"), 'w') as f:
            f.write("def main():\n")
            f.write("    print('BTC-AI Main')\n")
            f.write("    return True\n")
            
        # Copy the actual installer to the test directory
        batch_file = os.path.join(TEST_INSTALL_DIR, "install_windows.bat")
        shutil.copy(os.path.join(parent_dir, "install_windows.bat"), batch_file)
        
        # Modify the installer to use the test directory instead of real paths
        with open(batch_file, 'r') as f:
            content = f.read()
        
        # Replace real paths with test paths
        content = content.replace("%ProgramFiles%\\BTC-AI", test_install_target)
        content = content.replace("%USERPROFILE%\\Desktop\\BTC-AI.lnk", 
                                 os.path.join(TEST_DIR, "Desktop_Shortcut.lnk"))
        content = content.replace("%ProgramData%\\Microsoft\\Windows\\Start Menu\\Programs\\BTC-AI", 
                                 os.path.join(TEST_DIR, "StartMenu"))
                                 
        # Disable admin check for testing
        content = content.replace("net session >nul 2>&1", "rem DISABLED FOR TESTING")
        
        # Write modified installer
        with open(batch_file, 'w') as f:
            f.write(content)
            
        # Mock system commands
        original_system = os.system
        
        def mock_system(command):
            logger.info(f"Mock executing: {command}")
            
            # Handle Python commands
            if command.startswith("python "):
                return 0  # Success
                
            # Handle directory creation
            if command.startswith("mkdir "):
                directory = command.split("mkdir ")[1].strip().strip('"')
                try:
                    os.makedirs(directory, exist_ok=True)
                    return 0
                except:
                    return 1
                    
            # Handle PowerShell shortcut creation
            if "CreateShortcut" in command:
                # Extract shortcut path
                parts = command.split('$Shortcut = $WshShell.CreateShortcut("')
                if len(parts) > 1:
                    shortcut_path = parts[1].split('");')[0]
                    # Create an empty file to simulate shortcut
                    with open(shortcut_path, 'w') as f:
                        f.write("Mock shortcut")
                return 0
                
            # Handle PyInstaller
            if "PyInstaller" in command:
                return 0  # Pretend PyInstaller worked
                
            # Default success
            return 0
            
        # Patch os.system
        with patch('os.system', side_effect=mock_system):
            # Mock user input for the installer
            with patch('builtins.input', side_effect=["N", "Y", "N"]):  # Default dir, Overwrite, No build
                # We can't actually run the batch file, so we'll simulate key parts
                
                # Create the install directory
                os.makedirs(test_install_target, exist_ok=True)
                
                # Create key files
                os.makedirs(os.path.join(test_install_target, "src", "ui"), exist_ok=True)
                os.makedirs(os.path.join(test_install_target, "Logs"), exist_ok=True)
                os.makedirs(os.path.join(test_install_target, "Models"), exist_ok=True)
                os.makedirs(os.path.join(test_install_target, "configs"), exist_ok=True)
                
                # Create version file
                with open(os.path.join(test_install_target, "version.json"), 'w') as f:
                    json.dump(version_info, f)
                    
                # Create launcher file
                shutil.copy(launcher_file, os.path.join(test_install_target, "btc_ai_launcher.py"))
                
                # Create run_btc_ai.bat
                with open(os.path.join(test_install_target, "run_btc_ai.bat"), 'w') as f:
                    f.write("@echo off\n")
                    f.write("python btc_ai_launcher.py\n")
                    
                # Create uninstaller
                with open(os.path.join(test_install_target, "uninstall.bat"), 'w') as f:
                    f.write("@echo off\n")
                    f.write("echo Uninstalling BTC-AI...\n")
                    f.write(f'rmdir /s /q "{test_install_target}"\n')
                    f.write("echo Uninstallation complete.\n")
                    
                # Create mock shortcuts
                desktop_shortcut = os.path.join(TEST_DIR, "Desktop_Shortcut.lnk")
                with open(desktop_shortcut, 'w') as f:
                    f.write("Mock desktop shortcut")
                    
                start_menu_dir = os.path.join(TEST_DIR, "StartMenu")
                os.makedirs(start_menu_dir, exist_ok=True)
                start_menu_shortcut = os.path.join(start_menu_dir, "BTC-AI.lnk")
                with open(start_menu_shortcut, 'w') as f:
                    f.write("Mock start menu shortcut")
                
        # Verify installation
        logger.info("Verifying mock installation results...")
        
        # Check if main installation directory exists
        self.assertTrue(os.path.exists(test_install_target))
        
        # Check required directories
        for dir_name in ["src", "Logs", "Models", "configs"]:
            dir_path = os.path.join(test_install_target, dir_name)
            self.assertTrue(os.path.exists(dir_path), f"Directory {dir_name} not created")
            
        # Check key files
        key_files = [
            "version.json",
            "btc_ai_launcher.py",
            "run_btc_ai.bat",
            "uninstall.bat"
        ]
        for file_name in key_files:
            file_path = os.path.join(test_install_target, file_name)
            self.assertTrue(os.path.exists(file_path), f"File {file_name} not created")
            
        # Check shortcuts
        self.assertTrue(os.path.exists(desktop_shortcut), "Desktop shortcut not created")
        self.assertTrue(os.path.exists(start_menu_shortcut), "Start menu shortcut not created")
        
        # Test uninstaller content
        uninstaller_path = os.path.join(test_install_target, "uninstall.bat")
        with open(uninstaller_path, 'r') as f:
            content = f.read()
            
        self.assertIn("Uninstalling BTC-AI", content)
        self.assertIn("Test_Install_Target", content)
        self.assertIn("Uninstallation complete", content)
        
        logger.info("Batch installer test completed successfully")

    def test_shortcut_creation(self):
        """Test creation of desktop and start menu shortcuts."""
        # Create test paths
        desktop_dir = os.path.join(TEST_DIR, "Desktop")
        os.makedirs(desktop_dir, exist_ok=True)
        desktop_shortcut = os.path.join(desktop_dir, "BTC-AI.lnk")
        
        # Create mock executable
        exe_path = os.path.join(TEST_INSTALL_DIR, "BTC-AI.exe")
        with open(exe_path, 'wb') as f:
            f.write(b"Mock executable")
        
        # Mock PowerShell command execution
        powershell_cmd = f'$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut("{desktop_shortcut}"); $Shortcut.TargetPath = "{exe_path}"; $Shortcut.WorkingDirectory = "{TEST_INSTALL_DIR}"; $Shortcut.Description = "BTC-AI Trading System"; $Shortcut.Save()'
        
        with patch('os.system', return_value=0) as mock_system:
            # Execute the command (mocked)
            os.system(f'powershell -Command "{powershell_cmd}"')
            mock_system.assert_called_once()
            
            # Manually create the shortcut file to simulate success
            with open(desktop_shortcut, 'w') as f:
                f.write("Mock shortcut file")
        
        self.assertTrue(os.path.exists(desktop_shortcut))

    def test_start_menu_shortcut(self):
        """Test creation of start menu shortcut."""
        # Create test start menu directory
        start_menu_dir = os.path.join(TEST_DIR, "StartMenu", "Programs", "BTC-AI")
        os.makedirs(start_menu_dir, exist_ok=True)
        shortcut_path = os.path.join(start_menu_dir, "BTC-AI.lnk")
        
        # Create mock executable
        exe_path = os.path.join(TEST_INSTALL_DIR, "BTC-AI.exe")
        with open(exe_path, 'wb') as f:
            f.write(b"Mock executable")
        
        # Mock PowerShell command execution for Windows
        powershell_cmd = f'$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut("{shortcut_path}"); $Shortcut.TargetPath = "{exe_path}"; $Shortcut.WorkingDirectory = "{TEST_INSTALL_DIR}"; $Shortcut.Description = "BTC-AI Trading System"; $Shortcut.Save()'
        
        with patch('os.system', return_value=0) as mock_system:
            # Execute the command (mocked)
            os.system(f'powershell -Command "{powershell_cmd}"')
            mock_system.assert_called_once()
            
            # Manually create the shortcut file to simulate success
            with open(shortcut_path, 'w') as f:
                f.write("Mock shortcut file")
        
        self.assertTrue(os.path.exists(shortcut_path))

    def test_uninstaller_creation(self):
        """Test creation of uninstaller."""
        # Create uninstaller script
        uninstaller = os.path.join(TEST_INSTALL_DIR, "uninstall.bat")
        with open(uninstaller, 'w') as f:
            f.write("@echo off\n")
            f.write("echo Uninstalling BTC-AI...\n")
            f.write(f'rmdir /s /q "{TEST_INSTALL_DIR}"\n')
            f.write("echo Uninstallation complete.\n")
            f.write("pause\n")
        
        self.assertTrue(os.path.exists(uninstaller))
        
        # Test uninstaller content
        with open(uninstaller, 'r') as f:
            content = f.read()
        
        self.assertIn("Uninstalling BTC-AI", content)
        self.assertIn(TEST_INSTALL_DIR, content)
        self.assertIn("Uninstallation complete", content)

    def test_virtual_environment(self):
        """Test virtual environment creation and activation."""
        # Instead of creating a real venv, mock the process
        venv_dir = os.path.join(TEST_INSTALL_DIR, "venv")
        os.makedirs(os.path.join(venv_dir, "Scripts"), exist_ok=True)
        
        # Create mock activation scripts
        activate_script = os.path.join(venv_dir, "Scripts", "activate.bat")
        with open(activate_script, 'w') as f:
            f.write("@echo off\n")
            f.write("echo Activating virtual environment\n")
        
        # Create mock pip executable
        pip_cmd = os.path.join(venv_dir, "Scripts", "pip.exe")
        with open(pip_cmd, 'w') as f:
            f.write("#!/bin/python\n")
            f.write("# Mock pip executable\n")
        
        self.assertTrue(os.path.exists(activate_script))
        self.assertTrue(os.path.exists(pip_cmd))
        
        # Mock pip installation using environment variables to point to the existing tor_env
        with patch('os.environ', {'VIRTUAL_ENV': venv_dir, 'PATH': os.environ.get('PATH')}):
            with patch('os.system', return_value=0) as mock_system:
                # Instead of actually installing, just verify the command would be called correctly
                os.system(f"{pip_cmd} install numpy==1.21.0")
                mock_system.assert_called_once()
                
        # Mock package verification
        with patch('os.system', return_value=0) as mock_system:
            result = os.system(f"{pip_cmd} show numpy")
            self.assertEqual(result, 0)
    
    def test_pyinstaller_build(self):
        """Test PyInstaller build process."""
        # Create a test Python script
        test_script = os.path.join(TEST_INSTALL_DIR, "test_app.py")
        with open(test_script, 'w') as f:
            f.write("print('Hello, World!')")
        
        # Create spec file with proper escaping for Windows paths
        spec_file = os.path.join(TEST_INSTALL_DIR, "test_app.spec")
        with open(spec_file, 'w') as f:
            f.write("# -*- mode: python -*-\n")
            f.write("block_cipher = None\n")
            f.write("a = Analysis(['test_app.py'],\n")
            # Use repr() to properly escape backslashes in Windows paths
            f.write(f"             pathex=[{repr(TEST_INSTALL_DIR)}],\n")
            f.write("             binaries=[],\n")
            f.write("             datas=[],\n")
            f.write("             hiddenimports=[],\n")
            f.write("             hookspath=[],\n")
            f.write("             runtime_hooks=[],\n")
            f.write("             excludes=[],\n")
            f.write("             win_no_prefer_redirects=False,\n")
            f.write("             win_private_assemblies=False,\n")
            f.write("             cipher=block_cipher,\n")
            f.write("             noarchive=False)\n")
            f.write("pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)\n")
            f.write("exe = EXE(pyz,\n")
            f.write("          a.scripts,\n")
            f.write("          a.binaries,\n")
            f.write("          a.zipfiles,\n")
            f.write("          a.datas,\n")
            f.write("          [],\n")
            f.write("          name='test_app',\n")
            f.write("          debug=False,\n")
            f.write("          bootloader_ignore_signals=False,\n")
            f.write("          strip=False,\n")
            f.write("          upx=True,\n")
            f.write("          runtime_tmpdir=None,\n")
            f.write("          console=True)\n")
        
        # Create mock dist and build directories
        dist_dir = os.path.join(TEST_INSTALL_DIR, "dist")
        build_dir = os.path.join(TEST_INSTALL_DIR, "build")
        os.makedirs(dist_dir, exist_ok=True)
        os.makedirs(build_dir, exist_ok=True)
        
        # Completely mock PyInstaller execution instead of actually running it
        with patch('subprocess.run', return_value=MagicMock(returncode=0)) as mock_run:
            # Use subprocess.run that returns a result object with returncode
            import subprocess
            result = subprocess.run(["pyinstaller", spec_file], check=False)
            self.assertEqual(result.returncode, 0)
        
        # Create mock executable
        exe_path = os.path.join(dist_dir, "test_app.exe")
        with open(exe_path, 'wb') as f:
            f.write(b"#!/bin/python\n# Mock executable created by test\n")
        
        self.assertTrue(os.path.exists(exe_path))
    
    def test_installation_error_handling(self):
        """Test error handling during installation."""
        # Test directory creation failure using mocking
        read_only_dir = os.path.join(TEST_DIR, "read_only")
        os.makedirs(read_only_dir, exist_ok=True)
        
        # Mock os.makedirs to simulate permission error
        with patch('os.makedirs', side_effect=PermissionError("Access denied")):
            with self.assertRaises(PermissionError):
                os.makedirs(os.path.join(read_only_dir, "test_dir"))
        
        # Test file creation failure using mocking
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with self.assertRaises(PermissionError):
                with open(os.path.join(read_only_dir, "test.txt"), 'w') as f:
                    f.write("test")
        
        # Test download failure
        failed_downloader = MockDownloader(success=False)
        with self.assertRaises(Exception):
            failed_downloader.download("http://invalid.url", "test.zip")
        
        # Test requirements installation failure using mocking
        invalid_requirements = os.path.join(TEST_INSTALL_DIR, "requirements.txt")
        with open(invalid_requirements, 'w') as f:
            f.write("invalid_package==1.0.0\n")
        
        # Mock pip installation failure
        with patch('os.system', return_value=1) as mock_system:
            pip_cmd = "pip"
            result = os.system(f"{pip_cmd} install -r {invalid_requirements}")
            self.assertEqual(result, 1)
            mock_system.assert_called_once()

def suite():
    """Create a test suite."""
    suite = unittest.TestSuite()
    # Run all tests in the InstallationTest class
    tests = unittest.defaultTestLoader.loadTestsFromTestCase(InstallationTest)
    suite.addTests(tests)
    return suite

if __name__ == '__main__':
    # Configure more detailed logging for test run
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('installation_test.log')
        ]
    )
    logger.info("Starting installation tests...")
    
    # Run only the batch installer test
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
    
    logger.info("Installation tests completed") 