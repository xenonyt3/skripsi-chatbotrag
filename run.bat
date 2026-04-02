@echo off
title RAG Chatbot (BIARKAN CONSOLE OPEN!!)
set OLLAMA_MODELS=%~dp0.ollama\models
set OLLAMA_HOST=http://127.0.0.1:11434

start "" /b ollama serve
timeout /t 5 > nul

python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

pause