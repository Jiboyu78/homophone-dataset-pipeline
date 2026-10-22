# 会意猜谜数据集 pipeline 启动脚本（Windows 批处理）
@echo off
setlocal
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

set PY=D:\DSH\resources\runtime\primary-runtime\dependencies\python\python.exe
if not exist "%PY%" set PY=python

if "%~1"=="" (
  echo 用法: scripts\run.bat ^<子命令^> [参数]
  echo   示例: scripts\run.bat health
  echo         scripts\run.bat run --mock --limit 30
  exit /b 0
)

"%PY%" -m pipeline.cli %*
exit /b %ERRORLEVEL%
