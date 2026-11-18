@echo off
chcp 65001 >nul
setlocal
REM ---------------------------------------------------------------------------
REM  PointAnnotator portable launcher (ASCII-only on purpose)
REM
REM  Prefers a bundled interpreter at .\python\python.exe, otherwise falls
REM  back to python / py on PATH. A Python runtime IS required - this is not
REM  the "no Python needed" path, that is the PyInstaller build (build_exe.bat).
REM
REM  Usage: run.bat D:\videos
REM ---------------------------------------------------------------------------
cd /d "%~dp0"

if exist "%~dp0python\python.exe" goto :bundled
where python >nul 2>nul
if not errorlevel 1 goto :path_python
where py >nul 2>nul
if not errorlevel 1 goto :path_py
echo [ERROR] No Python found.
echo         Need .\python\python.exe or python / py on PATH.
echo         Install Python 3.10+ and tick "tcl/tk and IDLE" during setup.
exit /b 1

:bundled
"%~dp0python\python.exe" point_annotator.py %*
goto :end

:path_python
python point_annotator.py %*
goto :end

:path_py
py point_annotator.py %*

:end
if errorlevel 1 echo [ERROR] PointAnnotator exited with an error. See point_annotator.log
endlocal
