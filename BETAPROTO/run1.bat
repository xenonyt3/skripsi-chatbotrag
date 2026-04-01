@echo off
title RAG Chatbot (BIARKAN CONSOLE OPEN!)
set OLLAMA_HOME=%~dp0.ollama

start "" /b "%~dp0ollama\ollama.exe" serve
timeout /t 5 > nul

python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload

pause
