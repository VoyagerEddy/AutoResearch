@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 (
  echo.
  echo AutoResearch failed to start. Keep this window open and copy the error message.
  pause
)
