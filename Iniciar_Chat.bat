@echo off
cd /d "%~dp0"

if not exist venv (
    echo Ambiente virtual nao encontrado. Criando venv e instalando dependencias...
    python -m venv venv
    call venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

streamlit run app_chat.py
pause
