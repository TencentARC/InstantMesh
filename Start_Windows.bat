@echo off

REM Set Python to use UTF-8 encoding by default
set PYTHONUTF8=1

REM Set Python 3.10 installation path
set PYTHON="C:\Program Files\Python310\python.exe"

if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")
if not defined REQUIREMENTS_FILE (set "REQUIREMENTS_FILE=requirements.txt")
if not defined CONDA_INSTALL_DIR (set "CONDA_INSTALL_DIR=%UserProfile%\miniconda3")

set ERROR_REPORTING=TRUE
set CUDA_VERSION=12.1
set DISTUTILS_USE_SDK=1
set NVDIFFRAST_TORCH_LOAD_VERBOSE=1

REM Set VSPATH manually to the path of Visual Studio 2019 installation
set VSPATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community

REM Add Visual Studio 2019 paths
if defined VSPATH (
  set "INCLUDE=%VSPATH%\VC\Tools\MSVC\14.29.30133\include;%INCLUDE%"
  set "LIB=%VSPATH%\VC\Tools\MSVC\14.29.30133\lib\x64;%LIB%"
)

REM Add Windows SDK paths
set "INCLUDE=C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\ucrt;%INCLUDE%"
set "INCLUDE=C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\shared;%INCLUDE%"
set "INCLUDE=C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\um;%INCLUDE%"
set "LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\ucrt\x64;%LIB%"
set "LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\um\x64;%LIB%"

REM Add Python library path
set "PYTHON_LIB_PATH=C:\Program Files\Python310\libs"
set "LIB=%PYTHON_LIB_PATH%;%LIB%"

:start_venv
if ["%VENV_DIR%"] == ["-"] goto :skip_venv
if ["%SKIP_VENV%"] == ["1"] goto :skip_venv

dir "%VENV_DIR%\Scripts\Python.exe" >nul 2>&1
if %ERRORLEVEL% == 0 goto :activate_venv

echo Creating venv in directory %VENV_DIR% using python %PYTHON%
%PYTHON% -m venv "%VENV_DIR%"
if %ERRORLEVEL% == 0 (
    echo Virtual environment created successfully.
) else (
    echo Failed to create virtual environment.
    goto :end
)

:activate_venv
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
echo Activating venv...
call "%VENV_DIR%\Scripts\activate.bat"

REM Disable SSL verification and trust PyPI by default
set PYTHONHTTPSVERIFY=0
%PYTHON% -m pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org huggingface.co"

REM Check network connectivity within the venv
echo Checking network connectivity within the venv...
%PYTHON% -m pip install --timeout=30 --retries=3 -r %REQUIREMENTS_FILE%
if %ERRORLEVEL% == 0 (
    echo Network connectivity within the venv is OK.
) else (
    echo Network connectivity issue within the venv. Trying to install packages individually...
    for /f "delims=" %%p in (%REQUIREMENTS_FILE%) do (
        echo Installing package: %%p
        %PYTHON% -m pip install --timeout=30 --retries=3 %%p
        if !ERRORLEVEL! NEQ 0 (
            echo Failed to install package: %%p
            goto :end
        )
    )
)

:skip_venv
goto :launch

:launch
echo Launching script...
%PYTHON% app.py %*
pause
exit /b

:end
echo.
echo Launch unsuccessful. Exiting.
pause
