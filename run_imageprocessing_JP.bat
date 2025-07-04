@echo off
REM Activate venv and run Streamlit app

IF EXIST ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    streamlit run imageprocessing_JP.py
) ELSE (
    echo 仮想環境(.venv)が見つかりません。先にセットアップを行ってください。
    pause
)