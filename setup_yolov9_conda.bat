@echo off
REM ============================================================
REM  YOLOv9 - Anaconda environment setup script (Windows)
REM  Usage: open "Anaconda Prompt", cd into this folder, then run:
REM         setup_yolov9_conda.bat
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo [1/6] Creating conda environment "yolov9" (Python 3.9) ...
call conda create -n yolov9 python=3.9 -y
if errorlevel 1 (
    echo Failed to create environment. Make sure Anaconda/Miniconda is installed and on PATH.
    pause
    exit /b 1
)

echo.
echo [2/6] Activating environment ...
call conda activate yolov9
if errorlevel 1 (
    echo Failed to activate environment. Please run this script from "Anaconda Prompt", not a plain cmd window.
    pause
    exit /b 1
)

echo.
echo [3/6] Checking for NVIDIA GPU ...
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
    echo NVIDIA GPU detected. Installing CUDA-enabled PyTorch build ^(CUDA 11.8^) ...
    nvidia-smi
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo No NVIDIA GPU / driver detected. Installing CPU-only PyTorch build.
    echo If your PC actually has a dedicated GPU, install the NVIDIA driver first, then re-run this script.
    pip install torch torchvision torchaudio
)

echo.
echo [4/6] Downloading YOLOv9 source code ...
where git >nul 2>nul
if %errorlevel%==0 (
    if not exist yolov9 (
        git clone https://github.com/WongKinYiu/yolov9.git
    ) else (
        echo yolov9 folder already exists, skipping clone.
    )
) else (
    echo git not found. Please download the zip from https://github.com/WongKinYiu/yolov9
    echo extract it here, rename the folder to "yolov9", then re-run this script.
    pause
    exit /b 1
)

cd yolov9

echo.
echo [5/6] Installing YOLOv9 Python dependencies ...
pip install -r requirements.txt

echo.
echo [6/6] Downloading pretrained COCO weights ...
if not exist weights mkdir weights
cd weights
if not exist yolov9-c-converted.pt (
    curl -L -o yolov9-c-converted.pt https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt
)
if not exist gelan-c.pt (
    curl -L -o gelan-c.pt https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt
)
cd ..

echo.
echo ============================================================
echo  Setup complete!
echo  Test inference (inside the yolov9 folder, yolov9 env active):
echo.
echo    python detect.py --weights weights/yolov9-c-converted.pt --conf 0.25 --img 640 --device 0 --source data/images/horses.jpg
echo.
echo  No GPU? Change "--device 0" to "--device cpu"
echo.
echo  To use this environment in VS Code:
echo    1. Open VS Code, open this "yolov9" folder (File ^> Open Folder)
echo    2. Ctrl+Shift+P -^> "Python: Select Interpreter" -^> choose the "yolov9" conda env
echo    3. Install VS Code extensions: Python, Pylance (by Microsoft)
echo ============================================================
cd ..
pause
