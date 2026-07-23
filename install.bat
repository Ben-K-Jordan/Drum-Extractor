@echo off
REM Drum Extractor installer for Windows.
REM Run from the repo folder:  install.bat
REM (Windows equivalent of install.sh; if anything fails, follow the manual
REM  steps in the README.)

where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3) else (set PY=python)

echo ==^> Creating virtualenv at .venv
%PY% -m venv .venv || goto :fail

echo ==^> Upgrading pip
.venv\Scripts\python -m pip install --quiet --upgrade pip || goto :fail

echo ==^> Installing Drum Extractor (torch/demucs are large; this takes a while)
.venv\Scripts\pip install -e ".[all]" || goto :fail

echo ==^> Trying optional extras (safe to fail)
.venv\Scripts\pip install -e ".[adtof]" || echo   ADTOF skipped - the onset fallback will be used
.venv\Scripts\pip install -e ".[quantize]" || echo   madmom skipped - librosa fallback is used automatically

echo ==^> Creating launcher
> run.bat echo @echo off
>> run.bat echo "%~dp0.venv\Scripts\drum-extractor" web %%*

echo ==^> Checking the result
.venv\Scripts\drum-extractor doctor

echo.
echo Done. Start the app by double-clicking run.bat (the browser opens itself).
goto :eof

:fail
echo Install failed - see the message above, or follow the manual steps in the README.
exit /b 1
