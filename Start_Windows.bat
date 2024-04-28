@echo off

REM Set Python to use UTF-8 encoding by default
set PYTHONUTF8=1

if not defined PYTHON (
    for /f "delims=" %%P in ('where python') do (
        set PYTHON=%%P
    )
)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")
if not defined REQUIREMENTS_FILE (set "REQUIREMENTS_FILE=requirements.txt")
if not defined CONDA_INSTALL_DIR (set "CONDA_INSTALL_DIR=%UserProfile%\miniconda3")

REM You need to set this if nvdiffrast won't compile
REM Found in C:\Program Files\Microsoft Visual Studio\(TYPE OF STUDIO)\VC\Tools\MSVC\(VERSION#)\bin\Hostx64\x64
REM also in :find_msvcprt you need to set the 

set ERROR_REPORTING=TRUE
set CUDA_VERSION=12.1
set CUDA_DOWNLOAD_LINK=https://developer.download.nvidia.com/compute/cuda/12.4.0/network_installers/cuda_12.4.0_windows_network.exe
set MSVC_REDIST_URL=https://aka.ms/vs/17/release/vc_redist.x64.exe
set DISTUTILS_USE_SDK=1
set NVDIFFRAST_TORCH_LOAD_VERBOSE=1

REM Detect Visual Studio installation path
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"

if exist "%VSWHERE%" (
  for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set VSPATH=%%i
  )
)

if defined VSPATH (
  for /f "usebackq tokens=*" %%i in (`dir /b /s /o:n "%VSPATH%\VC\Tools\MSVC\*"`) do (
    set "MSVC_DIR=%%i"
    goto :found_msvc
  )
)

:found_msvc
if defined MSVC_DIR (
  set "INCLUDE=%MSVC_DIR%\include;%INCLUDE%"
  set "LIB=%MSVC_DIR%\lib\x64;%LIB%"
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

mkdir tmp 2>NUL

:start_venv
if ["%VENV_DIR%"] == ["-"] goto :skip_venv
if ["%SKIP_VENV%"] == ["1"] goto :skip_venv

dir "%VENV_DIR%\Scripts\Python.exe" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :activate_venv

for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
echo Creating venv in directory %VENV_DIR% using python %PYTHON_FULLNAME%
%PYTHON_FULLNAME% -m venv "%VENV_DIR%" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :activate_venv

echo Unable to create venv in directory "%VENV_DIR%"
goto :show_stdout_stderr

:activate_venv
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
echo Activating venv...
call "%VENV_DIR%\Scripts\activate.bat"

:install_msvc_redist
rem Check if Visual C++ Redistributable Packages are installed
where /r "%ProgramFiles%" vcruntime140.dll >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo Visual C++ Redistributable Packages are installed.
    goto :find_msvcprt
)

echo Downloading Visual C++ Redistributable Packages...
powershell -Command "Invoke-WebRequest %MSVC_REDIST_URL% -OutFile msvc_redist.exe"
echo Installing Visual C++ Redistributable Packages...
start /wait "" msvc_redist.exe /quiet
del msvc_redist.exe

:find_msvcprt
rem Find the location of msvcprt.lib in the x64 folder of Visual Studio
set "MSVC_LIB_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\lib\x64"
if exist "%MSVC_LIB_PATH%\msvcprt.lib" (
    goto :set_msvc_path
) else (
    echo Could not find msvcprt.lib in the expected location.
    goto :show_stdout_stderr
)

:set_msvc_path
set MSVC_PATH=%MSVC_LIB_PATH%

:check_cuda
%PYTHON% -c "import torch; print(torch.version.cuda)" >tmp/stdout.txt 2>tmp/stderr.txt
set /p INSTALLED_CUDA_VERSION=<tmp/stdout.txt
if "%INSTALLED_CUDA_VERSION%" == "%CUDA_VERSION%" (
    echo CUDA %CUDA_VERSION% is already installed.
    goto :install_requirements
)

:install_cuda
echo Downloading CUDA %CUDA_VERSION% Toolkit...
powershell -Command "Invoke-WebRequest %CUDA_DOWNLOAD_LINK% -OutFile cuda_installer.exe"
echo Installing CUDA %CUDA_VERSION% Toolkit...
start /wait "" cuda_installer.exe --silent --toolkit --noadmin
del cuda_installer.exe

set NVDIFFRAST_SOURCE_FILE_ENCODING=utf-8


:install_requirements
echo Installing requirements from %REQUIREMENTS_FILE%...
%PYTHON% -m pip install -r %REQUIREMENTS_FILE% --verbose
if %ERRORLEVEL% == 0 (
    echo Requirements installed successfully.
) else (
    echo Failed to install requirements.
    goto :show_stdout_stderr
)

:skip_venv	
goto :launch

:launch
echo Launching script...
%PYTHON% app.py %*
pause
exit /b

:show_stdout_stderr
echo.
echo exit code: %errorlevel%
for /f %%i in ("tmp\stdout.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stdout:
type tmp\stdout.txt

:show_stderr
for /f %%i in ("tmp\stderr.txt") do set size=%%~zi
if %size% equ 0 goto :endofscript
echo.
echo stderr:
type tmp\stderr.txt

:endofscript
echo.
echo Launch unsuccessful. Exiting.
pause
