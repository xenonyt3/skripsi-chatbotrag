@echo off
title  (BIARKAN CONSOLE OPEN!)
set OLLAMA_HOME=%~dp0.ollama

start "" /b "%~dp0ollama\ollama.exe" serve
timeout /t 5 > nul