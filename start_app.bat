@echo off
echo ==========================================
echo   Tag-Aware Photo Story Generator
echo ==========================================
echo.
echo Clearing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul

echo Starting Web App (using latest code)...
echo Press Ctrl+C to stop
echo.
python -B src\web_ui\app.py
pause
