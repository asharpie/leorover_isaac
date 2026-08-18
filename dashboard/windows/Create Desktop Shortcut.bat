@echo off
rem Puts a "Leo Mission Control" icon on your Desktop. Run once.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
 "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Leo Mission Control.lnk');" ^
 "$s.TargetPath='%~dp0Leo Mission Control.bat';" ^
 "$s.WorkingDirectory='%~dp0';" ^
 "$s.WindowStyle=7;" ^
 "$s.IconLocation='%%SystemRoot%%\System32\shell32.dll,18';" ^
 "$s.Description='Leo rover dashboard — starts everything and opens the app';" ^
 "$s.Save()"
echo Desktop shortcut created. Double-click "Leo Mission Control" on your Desktop.
pause
