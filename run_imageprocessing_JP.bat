@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run imageprocessing_JP.py

echo.
echo Streamlit app exited. Press any key to close.
pause > nul
