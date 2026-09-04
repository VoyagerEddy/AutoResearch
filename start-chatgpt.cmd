@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0chatgpt-tunnel.ps1" -Action Run
if errorlevel 1 (
  echo.
  echo ChatGPT connection failed. Run setup-chatgpt.cmd first, then review the message above.
  pause
)
