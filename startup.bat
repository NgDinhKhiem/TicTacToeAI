@echo off
setlocal

REM Get the current directory of this script
set "CURRENT_DIR=%~dp0"
REM Remove trailing backslash if present
if "%CURRENT_DIR:~-1%"=="\" set "CURRENT_DIR=%CURRENT_DIR:~0,-1%"

docker run -it --rm ^
    -p 8888:8888 ^
    -v "%CURRENT_DIR%:/app" ^
    -w /app ^
    tictactoe.ai /bin/bash

endlocal
