@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creando entorno virtual...
  py -3.12 -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install -r requirements.txt
)
start "PLC Anomaly Lab" http://127.0.0.1:5000
.venv\Scripts\python.exe app.py
endlocal
