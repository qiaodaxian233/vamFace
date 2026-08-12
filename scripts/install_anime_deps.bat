@echo off
setlocal EnableDelayedExpansion
REM ============================================================
REM  vamFace - optional anime deps installer (Windows)
REM  ASCII-only on purpose: immune to codepage/encoding issues.
REM  animeface is OPTIONAL - failure here never blocks vamFace.
REM ============================================================

echo ============================================
echo  vamFace anime deps installer (optional)
echo ============================================

REM ---- pick a Python: prefer 3.12 / 3.11 / 3.10 via the py launcher ----
set "PY="
for %%V in (3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "pass" >nul 2>nul && set "PY=py -%%V"
    )
)
if not defined PY set "PY=python"

for /f "tokens=2" %%I in ('%PY% --version 2^>^&1') do set "PYVER=%%I"
echo Using: %PY%  ^(Python %PYVER%^)

REM ---- warn on 3.13+ : compiled deps rarely have wheels for it ----
echo %PYVER% | findstr /b "3.13 3.14" >nul && (
    echo.
    echo [WARN] Python %PYVER% detected. Compiled deps of this project
    echo        ^(animeface now, insightface later for style=real^) often
    echo        have no prebuilt wheels for it on Windows.
    echo        Recommended: install Python 3.12, then re-run this script.
    echo.
)

REM ---- try animeface (needs a C compiler on Windows) ----
echo Trying: pip install animeface ...
%PY% -m pip install animeface
if %errorlevel% equ 0 (
    echo.
    echo [OK] animeface installed. style=anime will use landmark
    echo      geometry scoring.
    goto :done
)

echo.
echo [INFO] animeface failed to install. THIS IS FINE - it is optional:
echo.
echo   1. Skip it. style=anime degrades gracefully with a warning,
echo      vamFace keeps working.
echo   2. Better route on Windows - use your own anime face ONNX model:
echo        vamface-fit --style anime --anime-onnx path\to\model.onnx
echo      onnxruntime ships with the base install, nothing to compile.
echo   3. Last resort - install "Visual Studio Build Tools" with the
echo      "Desktop development with C++" workload, then re-run this:
echo        https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo      Note: ~7 GB, and animeface wraps a 2009-era C library, so the
echo      MSVC build may STILL fail. Route 2 is the reliable one.

:done
echo.
pause
