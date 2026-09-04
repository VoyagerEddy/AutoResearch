@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0chatgpt-tunnel.ps1" -Action Setup
if errorlevel 1 (
  echo.
  echo ChatGPT tunnel setup failed. Keep this window open and review the message above.
  pause
)
