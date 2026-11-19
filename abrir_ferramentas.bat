@echo off
REM Abre MLflow e Evidently UI no Windows (dois cliques neste arquivo).
REM Cada ferramenta roda numa janela propria: feche a janela para desligar.
chcp 65001 >nul
cd /d "%~dp0"
set MLFLOW_DISABLE_AGENT_HINT=1

if not exist mlflow.db (
    echo Primeira execucao: rodando o pipeline ^(cerca de 30 s^)...
    python -m src.pipeline
)

echo Iniciando MLflow em http://localhost:5000 ...
start "MLflow - nao feche esta janela" cmd /k python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

echo Iniciando Evidently UI em http://localhost:8001 ...
start "Evidently UI - nao feche esta janela" cmd /k python -c "import sys; sys.argv=['evidently','ui','--workspace','evidently_workspace','--port','8001']; from evidently.cli import app; app()"

echo Aguardando os servidores subirem...
timeout /t 20 /nobreak >nul

start "" http://localhost:5000/#/experiments
start "" http://localhost:8001

echo.
echo Pronto! MLflow: http://localhost:5000   Evidently UI: http://localhost:8001
echo Para desligar, feche as janelas "MLflow" e "Evidently UI".
pause
