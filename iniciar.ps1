param(
    [switch]$SemNavegador,
    [switch]$Reiniciar
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$logRoot = Join-Path $projectRoot 'resultados\local\logs'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Ambiente ausente. Siga a preparação inicial no README antes de abrir o projeto.'
}
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$settingsJson = & $pythonPath -c "import json,sys;sys.path.insert(0,sys.argv[1]);from drenavision.core.config import get_settings;s=get_settings();print(json.dumps({'port':s.api_port,'host':s.api_host}))" $projectRoot
if ($LASTEXITCODE -ne 0) { throw 'Dependências ausentes. Instale requirements-dev.txt conforme o README.' }
$runtimeConfig = $settingsJson | ConvertFrom-Json
$apiPort = [int]$runtimeConfig.port
$apiHost = [string]$runtimeConfig.host
$clientHost = if ($apiHost -eq '0.0.0.0') { '127.0.0.1' } else { $apiHost }
$healthUrl = "http://${clientHost}:$apiPort/health"

function Get-ProjectService([int]$Port, [string]$Module) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $listener) { return $null }
    $running = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    $command = [string]$running.CommandLine
    if (-not ($command.Contains($projectRoot) -and $command.Contains($Module))) {
        throw "A porta $Port está ocupada por outro programa. Feche esse programa ou libere a porta antes de iniciar."
    }
    return $running
}

function Start-ProjectService([int]$Port, [string]$Module, [string[]]$ServiceArguments, [string]$LogName) {
    $running = Get-ProjectService $Port $Module
    if ($running -and $Reiniciar) {
        Stop-Process -Id $running.ProcessId
        for ($attempt = 0; $attempt -lt 20; $attempt++) {
            if (-not (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)) { break }
            Start-Sleep -Milliseconds 250
        }
        $running = $null
    }
    if ($running) {
        Write-Host "Serviço já disponível na porta $Port."
        return
    }
    Start-Process -FilePath $pythonPath -ArgumentList $ServiceArguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logRoot "$LogName.out.log") -RedirectStandardError (Join-Path $logRoot "$LogName.err.log") | Out-Null
}

Start-ProjectService $apiPort 'uvicorn' @('-m', 'uvicorn', 'drenavision.api.main:app', '--app-dir', ('"' + $projectRoot + '"'), '--host', $apiHost, '--port', [string]$apiPort) 'api'
$uiPath = Join-Path $projectRoot 'drenavision\ui\streamlit_app.py'
Start-ProjectService 8501 'streamlit' @('-m', 'streamlit', 'run', ('"' + $uiPath + '"'), '--server.headless', 'true', '--server.address', '127.0.0.1', '--server.port', '8501') 'interface'

$ready = $false
for ($attempt = 0; $attempt -lt 45; $attempt++) {
    try {
        $health = Invoke-RestMethod $healthUrl -TimeoutSec 2
        $ui = Invoke-WebRequest 'http://127.0.0.1:8501/_stcore/health' -TimeoutSec 2
        if ($ui.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
    Start-Sleep -Milliseconds 500
}
if (-not $ready) { throw "Não foi possível iniciar. Veja os registros em resultados/local/logs/." }
if (-not $health.model_loaded) { Write-Warning 'A interface está pronta, mas falta o modelo. Veja no README como baixar best.pt.' }
Write-Host 'DrenaVision pronto: http://127.0.0.1:8501' -ForegroundColor Green
Write-Host "API técnica: http://${clientHost}:$apiPort/docs"
if (-not $SemNavegador) { Start-Process 'http://127.0.0.1:8501' }
