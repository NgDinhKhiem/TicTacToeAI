@echo off
REM Go to the parent folder of where this script is located
cd /d "%~dp0.."

REM Now run docker exec
docker exec -it ttt_ai /bin/bash
