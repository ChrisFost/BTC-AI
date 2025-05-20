@echo off
REM =============================================================
REM BTC-AI Windows Installation Script
REM =============================================================
REM This script installs the BTC-AI application for Windows
REM Future versions will include support for macOS and Linux
REM 
REM This installer uses a standalone executable approach that bundles
REM all required dependencies including PyTorch with CUDA support
REM if available, without requiring Conda or other external tools.
REM =============================================================

echo.
echo ========================================================
echo BTC-AI Installer for Windows
echo ========================================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo This installer needs to be run as Administrator.
    echo Please right-click on the file and select "Run as administrator".
    echo.
    pause
    exit /b 1
)

echo Checking system requirements...

REM Check Windows version
ver | find "10.0." > nul
if %errorLevel% neq 0 (
    echo WARNING: This application is optimized for Windows 10 or newer.
    echo You may experience compatibility issues.
    echo.
    timeout /t 3 > nul
)

REM Check if NVIDIA GPU and CUDA are available (optional)
set CUDA_AVAILABLE=0
where nvidia-smi >nul 2>&1
if %errorLevel% equ 0 (
    echo NVIDIA GPU detected - checking CUDA...
    nvidia-smi | find "CUDA Version" >nul 2>&1
    if %errorLevel% equ 0 (
        set CUDA_AVAILABLE=1
        for /f "tokens=6" %%i in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VERSION=%%i
        echo CUDA %CUDA_VERSION% detected. GPU acceleration will be enabled.
        
        REM Extract major version for PyTorch compatibility
        for /f "tokens=1 delims=." %%a in ("%CUDA_VERSION%") do set CUDA_MAJOR=%%a
        
        REM Map CUDA version to PyTorch CUDA toolkit version
        REM Reference mapping (subject to change with PyTorch updates):
        REM CUDA 12.x -> cu121, cu122, etc.
        REM CUDA 11.x -> cu118, cu117, cu116, etc.
        REM CUDA 10.x -> cu102, cu101, etc.
        if %CUDA_MAJOR% equ 12 (
            set TORCH_CUDA=cu121
        ) else if %CUDA_MAJOR% equ 11 (
            set TORCH_CUDA=cu118
        ) else if %CUDA_MAJOR% equ 10 (
            set TORCH_CUDA=cu102
        ) else (
            echo CUDA %CUDA_VERSION% may not be compatible with PyTorch.
            echo Will attempt to install with best match or fall back to CPU version.
            set TORCH_CUDA=cu118
        )
        
    ) else (
        echo NVIDIA GPU detected but CUDA not found.
        echo The application will run without GPU acceleration.
    )
) else (
    echo No NVIDIA GPU detected. The application will run in CPU-only mode.
)

REM Create application directory - using a completely self-contained approach
REM Default to Program Files, but allow customization to avoid system conflicts
set DEFAULT_INSTALL_DIR=%ProgramFiles%\BTC-AI
set INSTALL_DIR=%DEFAULT_INSTALL_DIR%

echo.
echo Default installation directory: %DEFAULT_INSTALL_DIR%
set /p CUSTOM_DIR=Use custom installation directory? (Y/N): 
if /i "%CUSTOM_DIR%" EQU "Y" (
    set /p INSTALL_DIR=Enter custom installation path: 
)

echo.
echo Installing to: %INSTALL_DIR%
if exist "%INSTALL_DIR%" (
    echo.
    echo Installation directory already exists.
    echo.
    set /p OVERWRITE=Overwrite existing installation? (Y/N): 
    if /i "%OVERWRITE%" neq "Y" (
        echo Installation canceled.
        pause
        exit /b 0
    )
    echo Removing existing installation...
    rmdir /s /q "%INSTALL_DIR%"
)

mkdir "%INSTALL_DIR%" 2>nul
if %errorLevel% neq 0 (
    echo Failed to create installation directory. Check permissions.
    pause
    exit /b 1
)

REM Create necessary subdirectories
echo Creating application directories...
mkdir "%INSTALL_DIR%\Models" 2>nul
mkdir "%INSTALL_DIR%\Logs" 2>nul
mkdir "%INSTALL_DIR%\Cache" 2>nul
mkdir "%INSTALL_DIR%\configs" 2>nul
mkdir "%INSTALL_DIR%\docs" 2>nul
mkdir "%INSTALL_DIR%\lib" 2>nul
mkdir "%INSTALL_DIR%\bin" 2>nul
mkdir "%INSTALL_DIR%\temp" 2>nul

REM Create version information file
echo Creating application version information...
echo { > "%INSTALL_DIR%\version.json"
echo   "version": "1.0.0", >> "%INSTALL_DIR%\version.json"
echo   "build_date": "%date%", >> "%INSTALL_DIR%\version.json"
echo   "python_version": "3.8.20", >> "%INSTALL_DIR%\version.json"
echo   "cuda_available": "%CUDA_AVAILABLE%", >> "%INSTALL_DIR%\version.json"
echo   "cuda_version": "%CUDA_VERSION%" >> "%INSTALL_DIR%\version.json"
echo } >> "%INSTALL_DIR%\version.json"

REM Check if we're installing from source or from a build
if exist "dist\BTC-AI" (
    echo Installing pre-built executable with all dependencies...
    xcopy /E /I /Y "dist\BTC-AI\*" "%INSTALL_DIR%" > nul
    
    REM Verify the executable was copied correctly
    if not exist "%INSTALL_DIR%\BTC-AI.exe" (
        echo ERROR: Failed to copy executable. Installation failed.
        pause
        exit /b 1
    )
    echo Executable successfully installed.
) else (
    REM This section is primarily for developers - normal users will get the pre-built version
    echo No pre-built executable found. This is unusual for end-user installation.
    echo.
    set /p CONTINUE_SRC=Continue with source installation (for developers only)? (Y/N): 
    if /i "%CONTINUE_SRC%" neq "Y" (
        echo Installation canceled.
        pause
        exit /b 0
    )
    
    echo Installing from source (DEVELOPER MODE)...
    
    REM Check Python installation (only needed for source installation)
    python --version > nul 2>&1
    if %errorLevel% neq 0 (
        echo ERROR: Python is not installed or not in PATH.
        echo Please install Python 3.8 or newer from https://www.python.org/downloads/
        echo.
        pause
        exit /b 1
    )
    
    REM Check Python version (exactly match 3.8.x if possible)
    for /f "tokens=2" %%I in ('python --version 2^>^&1') do set pyver=%%I
    for /f "tokens=1,2 delims=." %%a in ("%pyver%") do (
        set pymajor=%%a
        set pyminor=%%b
    )
    if %pymajor% LSS 3 (
        echo ERROR: Python version 3.8 or newer is required.
        echo Current version: %pyver%
        echo.
        pause
        exit /b 1
    )
    if %pymajor% EQU 3 (
        if %pyminor% LSS 8 (
            echo ERROR: Python version 3.8 or newer is required.
            echo Current version: %pyver%
            echo.
            pause
            exit /b 1
        )
        if %pyminor% GTR 8 (
            echo WARNING: This application was developed with Python 3.8.
            echo Current version: %pyver% (Compatibility may vary)
            echo.
            timeout /t 3 > nul
        )
    )
    
    echo Python %pyver% detected.
    
    REM Copy source files
    xcopy /E /I /Y "src" "%INSTALL_DIR%\src" > nul
    xcopy /E /I /Y "configs" "%INSTALL_DIR%\configs" > nul
    xcopy /E /I /Y "docs" "%INSTALL_DIR%\docs" > nul
    
    REM Copy required files
    copy "setup.py" "%INSTALL_DIR%" > nul
    copy "requirements.txt" "%INSTALL_DIR%" > nul
    copy "README.md" "%INSTALL_DIR%" > nul
    copy "USER_GUIDE.md" "%INSTALL_DIR%" > nul
    copy "btc_ai_launcher.py" "%INSTALL_DIR%" > nul
    
    REM Use existing requirements.txt if available, otherwise create one with specific versions
    if exist "requirements.txt" (
        echo Using existing requirements file...
        copy "requirements.txt" "%INSTALL_DIR%\requirements.txt" > nul
    ) else (
        echo Creating requirements file with specific versions...
        echo numpy==1.24.1 > "%INSTALL_DIR%\requirements.txt"
        echo pandas==2.0.3 >> "%INSTALL_DIR%\requirements.txt"
        echo matplotlib==3.7.5 >> "%INSTALL_DIR%\requirements.txt"
        echo seaborn==0.13.2 >> "%INSTALL_DIR%\requirements.txt"
        echo scikit-learn==1.3.2 >> "%INSTALL_DIR%\requirements.txt"
        echo scipy==1.10.1 >> "%INSTALL_DIR%\requirements.txt"
        echo PySimpleGUI==4.60.5 >> "%INSTALL_DIR%\requirements.txt"
        echo tqdm==4.67.1 >> "%INSTALL_DIR%\requirements.txt"
        echo stable-baselines3==2.4.1 >> "%INSTALL_DIR%\requirements.txt"
        echo gymnasium==1.0.0 >> "%INSTALL_DIR%\requirements.txt"
        echo gym==0.26.2 >> "%INSTALL_DIR%\requirements.txt"
        echo psutil==6.1.1 >> "%INSTALL_DIR%\requirements.txt"
        echo pillow==10.2.0 >> "%INSTALL_DIR%\requirements.txt"
        echo tensorboardx==2.6.2.2 >> "%INSTALL_DIR%\requirements.txt"
        echo pyinstaller==6.12.0 >> "%INSTALL_DIR%\requirements.txt"
    )
    
    REM Create a completely isolated virtual environment 
    REM This doesn't modify system paths and can include special handling for conda-forge packages
    echo Creating isolated Python environment (no system modifications)...
    python -m venv "%INSTALL_DIR%\venv" --system-site-packages
    if %errorLevel% neq 0 (
        echo WARNING: Failed to create virtual environment.
        echo Attempting alternative approach...
        python -m pip install virtualenv --target="%INSTALL_DIR%\lib\python"
        set PYTHONPATH=%INSTALL_DIR%\lib\python
        python -m virtualenv "%INSTALL_DIR%\venv"
        if %errorLevel% neq 0 (
            echo WARNING: Virtual environment creation failed.
            echo The application will use a local library directory approach.
            set USE_VENV=0
        ) else {
            set USE_VENV=1
        }
    ) else (
        set USE_VENV=1
    )
    
    if %USE_VENV% equ 1 (
        REM Install dependencies in the virtual environment
        echo Installing dependencies in isolated environment...
        call "%INSTALL_DIR%\venv\Scripts\activate.bat"
        
        REM Create a special pip.ini to allow for both PyPI and conda-forge sources
        mkdir "%INSTALL_DIR%\venv\pip" 2>nul
        echo [global] > "%INSTALL_DIR%\venv\pip\pip.ini"
        echo extra-index-url = https://conda.anaconda.org/conda-forge/win-64 >> "%INSTALL_DIR%\venv\pip\pip.ini"
        echo                    https://conda.anaconda.org/conda-forge/noarch >> "%INSTALL_DIR%\venv\pip\pip.ini"
        
        set PIP_CONFIG_FILE=%INSTALL_DIR%\venv\pip\pip.ini
        
        python -m pip install --upgrade pip
        python -m pip install --no-cache-dir -r "%INSTALL_DIR%\requirements.txt"
        
        REM Install PyTorch with appropriate CUDA support
        if %CUDA_AVAILABLE% equ 1 (
            echo Installing PyTorch 2.4.1 with CUDA support (%TORCH_CUDA%)...
            python -m pip install --no-cache-dir torch==2.4.1+%TORCH_CUDA% torchvision==0.19.1+%TORCH_CUDA% torchaudio==2.4.1+%TORCH_CUDA% --index-url https://download.pytorch.org/whl/%TORCH_CUDA%
            
            REM Verify CUDA is actually available in PyTorch
            python -c "import torch; print('CUDA Available:', torch.cuda.is_available())" > "%INSTALL_DIR%\temp\cuda_check.txt"
            type "%INSTALL_DIR%\temp\cuda_check.txt" | find "CUDA Available: True" >nul
            if %errorLevel% neq 0 (
                echo WARNING: PyTorch with CUDA was installed but CUDA is not available.
                echo Falling back to CPU version...
                python -m pip uninstall -y torch torchvision torchaudio
                python -m pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
            ) else (
                echo PyTorch with CUDA support successfully installed and verified.
            )
        ) else (
            echo Installing PyTorch 2.4.1 (CPU version)...
            python -m pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
        )
        
        REM Special handling for packages that might need conda-forge
        echo Handling potentially problematic packages...
        
        REM Try to install TA-Lib if available
        echo Attempting to install TA-Lib (optional)...
        python -m pip install --no-cache-dir ta-lib==0.4.19 >nul 2>&1
        if %errorLevel% neq 0 (
            echo WARNING: Standard TA-Lib installation failed, trying alternative sources...
            python -m pip install --no-cache-dir ta-lib==0.4.19 --extra-index-url https://conda.anaconda.org/conda-forge/win-64 >nul 2>&1
            if %errorLevel% neq 0 (
                echo WARNING: TA-Lib installation failed. Some technical analysis features may be unavailable.
                echo TA-Lib can be installed manually later if needed.
            ) else (
                echo TA-Lib successfully installed from conda-forge.
            )
        ) else (
            echo TA-Lib successfully installed.
        )
        
        REM Finally, create a launch script that sets up the correct environment
        echo Creating environment launcher...
        echo @echo off > "%INSTALL_DIR%\run_btc_ai.bat"
        echo REM This script ensures the application runs in its isolated environment >> "%INSTALL_DIR%\run_btc_ai.bat"
        echo call "%INSTALL_DIR%\venv\Scripts\activate.bat" >> "%INSTALL_DIR%\run_btc_ai.bat"
        echo python "%INSTALL_DIR%\btc_ai_launcher.py" %%* >> "%INSTALL_DIR%\run_btc_ai.bat"
        echo call "%INSTALL_DIR%\venv\Scripts\deactivate.bat" >> "%INSTALL_DIR%\run_btc_ai.bat"
        
        REM Deactivate the virtual environment
        call "%INSTALL_DIR%\venv\Scripts\deactivate.bat"
    ) else (
        REM Alternative approach for systems where venv doesn't work
        REM Uses a local lib directory instead
        echo Using local library approach (no virtual environment)...
        mkdir "%INSTALL_DIR%\lib\python" 2>nul
        
        REM Install packages to local directory
        echo Installing dependencies to local directory...
        python -m pip install --target="%INSTALL_DIR%\lib\python" --upgrade pip
        python -m pip install --target="%INSTALL_DIR%\lib\python" --no-cache-dir -r "%INSTALL_DIR%\requirements.txt"
        
        REM Install PyTorch locally
        if %CUDA_AVAILABLE% equ 1 (
            echo Installing PyTorch 2.4.1 with CUDA support (%TORCH_CUDA%) to local directory...
            python -m pip install --target="%INSTALL_DIR%\lib\python" --no-cache-dir torch==2.4.1+%TORCH_CUDA% torchvision==0.19.1+%TORCH_CUDA% torchaudio==2.4.1+%TORCH_CUDA% --index-url https://download.pytorch.org/whl/%TORCH_CUDA%
        ) else (
            echo Installing PyTorch 2.4.1 (CPU version) to local directory...
            python -m pip install --target="%INSTALL_DIR%\lib\python" --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
        )
        
        REM Create a launcher script that sets PYTHONPATH
        echo Creating environment launcher...
        echo @echo off > "%INSTALL_DIR%\run_btc_ai.bat"
        echo REM This script ensures the application uses the local Python libraries >> "%INSTALL_DIR%\run_btc_ai.bat"
        echo set ORIGINAL_PYTHONPATH=%%PYTHONPATH%% >> "%INSTALL_DIR%\run_btc_ai.bat"
        echo set PYTHONPATH=%INSTALL_DIR%\lib\python;%%PYTHONPATH%% >> "%INSTALL_DIR%\run_btc_ai.bat"
        echo python "%INSTALL_DIR%\btc_ai_launcher.py" %%* >> "%INSTALL_DIR%\run_btc_ai.bat"
        echo set PYTHONPATH=%%ORIGINAL_PYTHONPATH%% >> "%INSTALL_DIR%\run_btc_ai.bat"
    )
    
    REM Build the application (optional)
    echo.
    set /p BUILD_APP=Build standalone executable? (Y/N): 
    if /i "%BUILD_APP%" EQU "Y" (
        echo Building application (this may take a while)...
        
        if %USE_VENV% equ 1 (
            call "%INSTALL_DIR%\venv\Scripts\activate.bat"
            
            REM Create PyInstaller spec file with safe defaults
            echo Creating PyInstaller specification...
            echo # -*- mode: python -*- > "%INSTALL_DIR%\btc_ai.spec"
            echo block_cipher = None >> "%INSTALL_DIR%\btc_ai.spec"
            echo a = Analysis(['btc_ai_launcher.py'], >> "%INSTALL_DIR%\btc_ai.spec"
            echo              pathex=['%INSTALL_DIR%'], >> "%INSTALL_DIR%\btc_ai.spec"
            echo              binaries=[], >> "%INSTALL_DIR%\btc_ai.spec"
            echo              datas=[('configs', 'configs'), ('docs', 'docs')], >> "%INSTALL_DIR%\btc_ai.spec"
            echo              hiddenimports=[], >> "%INSTALL_DIR%\btc_ai.spec"
            echo              hookspath=[], >> "%INSTALL_DIR%\btc_ai.spec"
            echo              runtime_hooks=[], >> "%INSTALL_DIR%\btc_ai.spec"
            echo              excludes=[], >> "%INSTALL_DIR%\btc_ai.spec"
            echo              win_no_prefer_redirects=False, >> "%INSTALL_DIR%\btc_ai.spec"
            echo              win_private_assemblies=False, >> "%INSTALL_DIR%\btc_ai.spec"
            echo              cipher=block_cipher, >> "%INSTALL_DIR%\btc_ai.spec"
            echo              noarchive=False) >> "%INSTALL_DIR%\btc_ai.spec"
            echo pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher) >> "%INSTALL_DIR%\btc_ai.spec"
            echo exe = EXE(pyz, >> "%INSTALL_DIR%\btc_ai.spec"
            echo          a.scripts, >> "%INSTALL_DIR%\btc_ai.spec"
            echo          a.binaries, >> "%INSTALL_DIR%\btc_ai.spec"
            echo          a.zipfiles, >> "%INSTALL_DIR%\btc_ai.spec"
            echo          a.datas, >> "%INSTALL_DIR%\btc_ai.spec"
            echo          [], >> "%INSTALL_DIR%\btc_ai.spec"
            echo          name='BTC-AI', >> "%INSTALL_DIR%\btc_ai.spec"
            echo          debug=False, >> "%INSTALL_DIR%\btc_ai.spec"
            echo          bootloader_ignore_signals=False, >> "%INSTALL_DIR%\btc_ai.spec"
            echo          strip=False, >> "%INSTALL_DIR%\btc_ai.spec"
            echo          upx=True, >> "%INSTALL_DIR%\btc_ai.spec"
            echo          runtime_tmpdir=None, >> "%INSTALL_DIR%\btc_ai.spec"
            echo          console=False) >> "%INSTALL_DIR%\btc_ai.spec"
            
            cd "%INSTALL_DIR%" && python -m PyInstaller --clean btc_ai.spec
            call "%INSTALL_DIR%\venv\Scripts\deactivate.bat"
        ) else (
            set PYTHONPATH=%INSTALL_DIR%\lib\python;%PYTHONPATH%
            cd "%INSTALL_DIR%" && python -m PyInstaller --clean --onedir --name=BTC-AI btc_ai_launcher.py
        )
        
        if exist "%INSTALL_DIR%\dist\BTC-AI" (
            echo Build successful!
            echo Standalone executable created: %INSTALL_DIR%\dist\BTC-AI\BTC-AI.exe
            
            REM Copy the executable back to the main directory
            copy "%INSTALL_DIR%\dist\BTC-AI\BTC-AI.exe" "%INSTALL_DIR%\BTC-AI.exe" > nul
        ) else (
            echo Build failed, continuing with source installation.
        )
    )
)

REM Create desktop shortcut
echo Creating shortcuts...
set SHORTCUT_PATH=%USERPROFILE%\Desktop\BTC-AI.lnk

if exist "%INSTALL_DIR%\BTC-AI.exe" (
    REM Create shortcut to EXE
    powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%INSTALL_DIR%\BTC-AI.exe'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.IconLocation = '%INSTALL_DIR%\BTC-AI.exe,0'; $Shortcut.Description = 'BTC-AI Trading System'; $Shortcut.Save()"
    
    echo Created shortcut to standalone executable.
) else if exist "%INSTALL_DIR%\run_btc_ai.bat" (
    REM Create shortcut to the batch launcher
    powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%INSTALL_DIR%\run_btc_ai.bat'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'BTC-AI Trading System'; $Shortcut.Save()"
    
    echo Created shortcut to environment launcher.
) else if defined USE_VENV (
    REM Create shortcut to Python script using virtual environment
    set VENV_PYTHON=%INSTALL_DIR%\venv\Scripts\pythonw.exe
    powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%VENV_PYTHON%'; $Shortcut.Arguments = '%INSTALL_DIR%\btc_ai_launcher.py'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'BTC-AI Trading System'; $Shortcut.Save()"
    
    echo Created shortcut using virtual environment.
) else (
    REM Create shortcut to Python script
    powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = 'pythonw'; $Shortcut.Arguments = '%INSTALL_DIR%\btc_ai_launcher.py'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'BTC-AI Trading System'; $Shortcut.Save()"
    
    echo Created shortcut using system Python.
)

REM Create start menu shortcut
set START_MENU_DIR=%ProgramData%\Microsoft\Windows\Start Menu\Programs\BTC-AI
mkdir "%START_MENU_DIR%" 2>nul
set START_MENU_SHORTCUT=%START_MENU_DIR%\BTC-AI.lnk

if exist "%INSTALL_DIR%\BTC-AI.exe" (
    REM Create shortcut to EXE
    powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%START_MENU_SHORTCUT%'); $Shortcut.TargetPath = '%INSTALL_DIR%\BTC-AI.exe'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.IconLocation = '%INSTALL_DIR%\BTC-AI.exe,0'; $Shortcut.Description = 'BTC-AI Trading System'; $Shortcut.Save()"
) else if exist "%INSTALL_DIR%\run_btc_ai.bat" (
    REM Create shortcut to the batch launcher
    powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%START_MENU_SHORTCUT%'); $Shortcut.TargetPath = '%INSTALL_DIR%\run_btc_ai.bat'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'BTC-AI Trading System'; $Shortcut.Save()"
) else if defined USE_VENV (
    REM Create shortcut to Python script using virtual environment
    set VENV_PYTHON=%INSTALL_DIR%\venv\Scripts\pythonw.exe
    powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%START_MENU_SHORTCUT%'); $Shortcut.TargetPath = '%VENV_PYTHON%'; $Shortcut.Arguments = '%INSTALL_DIR%\btc_ai_launcher.py'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'BTC-AI Trading System'; $Shortcut.Save()"
) else (
    REM Create shortcut to Python script
    powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%START_MENU_SHORTCUT%'); $Shortcut.TargetPath = 'pythonw'; $Shortcut.Arguments = '%INSTALL_DIR%\btc_ai_launcher.py'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'BTC-AI Trading System'; $Shortcut.Save()"
)

REM Create uninstaller
echo Creating uninstaller...
set UNINSTALLER=%INSTALL_DIR%\uninstall.bat
echo @echo off > "%UNINSTALLER%"
echo echo Uninstalling BTC-AI... >> "%UNINSTALLER%"
echo rmdir /s /q "%INSTALL_DIR%" >> "%UNINSTALLER%"
echo del "%SHORTCUT_PATH%" >> "%UNINSTALLER%"
echo rmdir /s /q "%START_MENU_DIR%" >> "%UNINSTALLER%"
echo echo Uninstallation complete. >> "%UNINSTALLER%"
echo pause >> "%UNINSTALLER%"

REM Create uninstaller shortcut in Start Menu
set UNINSTALL_SHORTCUT=%START_MENU_DIR%\Uninstall BTC-AI.lnk
powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%UNINSTALL_SHORTCUT%'); $Shortcut.TargetPath = '%UNINSTALLER%'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'Uninstall BTC-AI'; $Shortcut.Save()"

echo.
echo ========================================================
echo Installation Complete!
echo ========================================================
echo.
echo The BTC-AI application has been successfully installed.
if %CUDA_AVAILABLE% equ 1 (
    echo GPU acceleration is enabled with CUDA %CUDA_VERSION%.
) else (
    echo The application will run in CPU-only mode.
)
echo.
echo IMPORTANT: This installation is completely self-contained and does not modify
echo any system paths or settings outside of its own directory. The application
echo environment is isolated and will only run while the application is running.
echo.
echo Shortcuts have been created on your desktop and in the Start Menu.
echo.
echo Note: Future versions will support macOS and Linux installations.
echo.
pause 