@echo off
cd /d "%~dp0"
if not exist config.json copy config.example.json config.json >nul
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
