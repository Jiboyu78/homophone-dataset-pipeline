@echo off
chcp 65001 >nul
title Rebus Dataset Generator
cd /d "%~dp0.."
echo.
echo   ==========================================================
echo    Chinese Rebus Image Dataset  -  one click
echo   ==========================================================
echo.
echo    It will ask for two models' API key and base URL:
echo      1) description model  (pick words, design the puzzle,
echo                             write the image prompt)
echo      2) image model        (text to image)
echo.
echo    Everything else is automatic.
echo.
echo    Tip: pass flags through, e.g.
echo      scripts\生成数据集.bat --limit 200 --reuse
echo      scripts\生成数据集.bat --llm-batch 1     (if your gateway 504s)
echo      scripts\生成数据集.bat --help
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0oneclick.ps1" %*
echo.
pause
