@echo off

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")
if not defined REQUIREMENTS_FILE (set "REQUIREMENTS_FILE=requirements.txt")
if not defined CONDA_URL (set "CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe")
if not defined CONDA_INSTALL_DIR (set "CONDA_INSTALL_DIR=%UserProfile%\miniconda3")

set ERROR_REPORTING=TRUE

mkdir tmp 2>NUL

where conda >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_cuda

echo Conda not found. Installing Conda...
powershell -Command "Invoke-WebRequest -Uri '%CONDA_URL%' -OutFile 'miniconda_installer.exe'"
start /wait "" miniconda_installer.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%CONDA_INSTALL_DIR%
set "PATH=%CONDA_INSTALL_DIR%;%CONDA_INSTALL_DIR%\Scripts;%CONDA_INSTALL_DIR%\Library\bin;%PATH%"
del miniconda_installer.exe

:install_cuda
echo Installing CUDA using conda...
conda install -y cuda -c nvidia/label/cuda-12.1.0
if %ERRORLEVEL% == 0 (
    echo CUDA installed successfully.
) else (
    echo Failed to install CUDA using conda.
)

:install_requirements
%PYTHON% -c "" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_pip

echo Couldn't launch python
goto :show_stdout_stderr

:check_pip
%PYTHON% -mpip --help >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :start_venv

if "%PIP_INSTALLER_LOCATION%" == "" goto :show_stdout_stderr
%PYTHON% "%PIP_INSTALLER_LOCATION%" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :start_venv

echo Couldn't install pip
goto :show_stdout_stderr

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

:install_requirements
echo Installing requirements from %REQUIREMENTS_FILE%...
%PYTHON% -m pip install -r %REQUIREMENTS_FILE% --verbose
if %ERRORLEVEL% == 0 (
    echo Requirements installed successfully.
    goto :launch
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
