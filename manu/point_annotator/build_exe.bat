@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
REM ---------------------------------------------------------------------------
REM  Build the Windows distribution with PyInstaller (ASCII-only on purpose)
REM
REM  Output: dist\PointAnnotator\PointAnnotator.exe
REM  Copy the WHOLE dist\PointAnnotator folder to the annotator machine -
REM  the target machine does NOT need Python.
REM
REM  --onedir is used instead of --onefile on purpose: onefile unpacks the
REM  whole runtime into a temp dir on every start (slow) and is often
REM  flagged by antivirus software.
REM ---------------------------------------------------------------------------
cd /d "%~dp0"

echo [1/3] Checking Python ...
set "PY="
where python >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto :check
where py >nul 2>nul
if not errorlevel 1 set "PY=py"
if defined PY goto :check
echo [ERROR] No Python found.
echo         Install Python 3.10+ and add it to PATH, then run this again.
exit /b 1

:check
%PY% -V
echo [2/3] Installing / upgrading PyInstaller ...
%PY% -m pip install --upgrade pyinstaller
if errorlevel 1 goto :pipfail
echo [3/3] Building ...
if exist "%~dp0build" rmdir /s /q "%~dp0build"
if exist "%~dp0dist" rmdir /s /q "%~dp0dist"
%PY% -m PyInstaller --onedir --windowed --name PointAnnotator --clean --noconfirm point_annotator.py
if errorlevel 1 goto :buildfail
echo.
echo [OK] Output folder : %~dp0dist\PointAnnotator
echo [OK] Executable   : %~dp0dist\PointAnnotator\PointAnnotator.exe
echo.
echo Copy the WHOLE dist\PointAnnotator folder to the annotator machine.
echo The target machine does NOT need Python.
endlocal
exit /b 0

:pipfail
echo [ERROR] PyInstaller install failed. Check the network and try again.
endlocal
exit /b 1

:buildfail
echo [ERROR] Build failed. See the output above.
endlocal
exit /b 1
