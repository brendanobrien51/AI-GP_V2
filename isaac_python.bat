@echo off
setlocal

:: Isaac Sim standalone environment setup
set CARB_APP_PATH=C:\isaacsim\kit
set ISAAC_PATH=C:\isaacsim
set EXP_PATH=C:\isaacsim\apps
set OMNI_KIT_ACCEPT_EULA=YES
set PYTHONPATH=C:\isaacsim\python_packages;%PYTHONPATH%

:: Use the standalone's bundled Python
"C:\isaacsim\kit\python\python.exe" %*
