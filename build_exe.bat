@echo off
REM Build AACO as a standalone Windows executable
REM
REM Usage:
REM     build_exe.bat
REM
REM Output:
REM     dist\aaco.exe

echo ============================================
echo   Building AACO Executable
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

REM Check if PyInstaller is installed
python -c "import PyInstaller" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Install dependencies
echo Installing dependencies...
pip install -e . --quiet

REM Clean previous build
if exist "dist" (
    echo Cleaning previous build...
    rmdir /s /q dist
)
if exist "build" (
    rmdir /s /q build
)

REM Build the executable
echo Building executable...
pyinstaller aaco.spec --clean --noconfirm

REM Check if build succeeded
if exist "dist\aaco.exe" (
    echo.
    echo ============================================
    echo   Build Successful!
    echo ============================================
    echo.
    echo Executable: dist\aaco.exe
    echo.
    echo Usage examples:
    echo   dist\aaco.exe --help
    echo   dist\aaco.exe run model.onnx --backend cpu --iterations 50
    echo   dist\aaco.exe report ./aaco_sessions/latest
    echo.
) else (
    echo.
    echo ERROR: Build failed!
    exit /b 1
)
