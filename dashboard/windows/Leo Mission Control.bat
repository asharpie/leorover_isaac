@echo off
rem Leo Mission Control launcher.
rem First run (no key yet): shows a window for the one-time password step.
rem Every run after that: fully silent — tunnel + app window, no terminals.
if exist "%USERPROFILE%\.ssh\leo_dashboard" (
  powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0LeoMissionControl.ps1"
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0LeoMissionControl.ps1"
  pause
)
