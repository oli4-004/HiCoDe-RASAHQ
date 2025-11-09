# -------------------------------------------------
# CampusCompass\run-and-train-dev.ps1
# 1) Kill oude processen
# 2) Train Rasa model
# 3) Start action server (5055) + Rasa server (5005)
# 4) Wacht tot Rasa draait
# 5) Start frontend (5500)
# -------------------------------------------------

$ErrorActionPreference = "Continue"

# Paths
$BOT_ROOT     = $PSScriptRoot
$PROJECT_ROOT = Split-Path $BOT_ROOT -Parent
$VENV_PYTHON  = Join-Path $BOT_ROOT ".venv\Scripts\python.exe"
$VENV_RASA    = Join-Path $BOT_ROOT ".venv\Scripts\rasa.exe"

# Logs
$LOG_DIR = Join-Path $BOT_ROOT "logs"
if (!(Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Path $LOG_DIR | Out-Null
}
$actionsLog = Join-Path $LOG_DIR "actions_server.log"
$coreLog    = Join-Path $LOG_DIR "rasa_core.log"

Write-Host ("BOT_ROOT       = {0}" -f $BOT_ROOT)
Write-Host ("PROJECT_ROOT   = {0}" -f $PROJECT_ROOT)
Write-Host ("VENV_PYTHON    = {0}" -f $VENV_PYTHON)
Write-Host ("VENV_RASA      = {0}" -f $VENV_RASA)
Write-Host ("LOG_DIR        = {0}" -f $LOG_DIR)

if (!(Test-Path $VENV_PYTHON)) {
    Write-Host "ERROR: No venv found at $VENV_PYTHON"
    Write-Host "Create venv and pip install -r requirements.txt first."
    exit 1
}

Write-Host "venv OK"

function Kill-Port {
    param([int]$port)

    $conns = netstat -ano | Select-String (":$port")
    foreach ($line in $conns) {
        $parts = $line.ToString().Trim() -split "\s+"
        if ($parts.Length -ge 5) {
            $pid_train = $parts[-1]
            taskkill /PID $pid_train /F 2>$null | Out-Null
            Write-Host ("killed PID {0} on port {1}" -f $pid_train, $port)
        }
    }
}

function Wait-For-Rasa {
    param(
        [string]$Url = "http://localhost:5005/status",
        [int]$TimeoutSeconds = 40
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -Method Get -TimeoutSec 5
            if ($r.StatusCode -eq 200) {
                Write-Host "Rasa is up at $Url"
                return $true
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }
    Write-Host "Rasa did not become ready within $TimeoutSeconds seconds."
    return $false
}

# 1) Kill oude processen
Kill-Port 5005
Kill-Port 5055
Kill-Port 5500

# 2) Train
Write-Host ""
Write-Host "Training Rasa model ..."
Set-Location $BOT_ROOT

& $VENV_RASA train `
    --domain domain.yml `
    --data data `
    --config config.yml `
    --out models `
    --fixed-model-name dev

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Rasa training failed (exit $LASTEXITCODE)."
    exit $LASTEXITCODE
}

Write-Host "Training complete. Model stored in /models."
Write-Host ""

# 3) Start ACTION SERVER
Write-Host "starting action server job on :5055 ..."
$actionsJob = Start-Job -ScriptBlock {
    param($PROJECT_ROOT, $VENV_RASA, $actionsLog)
    Set-Location $PROJECT_ROOT
    & $VENV_RASA run actions `
        --actions CampusCompass.app.actions.actions `
        --port 5055 `
        *> $actionsLog
} -ArgumentList $PROJECT_ROOT, $VENV_RASA, $actionsLog
Write-Host ("actions job ID = {0}" -f $actionsJob.Id)

Start-Sleep -Seconds 2

# 4) Start RASA SERVER
Write-Host "starting rasa core job on :5005 ..."
$coreJob = Start-Job -ScriptBlock {
    param($BOT_ROOT, $VENV_RASA, $coreLog)
    Set-Location $BOT_ROOT
    & $VENV_RASA run `
        --enable-api `
        --cors "*" `
        --port 5005 `
        --credentials credentials.yml `
        --endpoints endpoints.yml `
        --model models `
        *> $coreLog
} -ArgumentList $BOT_ROOT, $VENV_RASA, $coreLog
Write-Host ("core job ID    = {0}" -f $coreJob.Id)

# 5) Wacht tot Rasa draait
Write-Host "Waiting for Rasa to become ready..."
$ready = Wait-For-Rasa
if (-not $ready) {
    Write-Host "ERROR: Rasa not ready. Check logs in $LOG_DIR."
}

# 6) Start frontend
Write-Host "starting frontend http://localhost:5500 ..."
$WEB_ROOT = Join-Path $BOT_ROOT "web"
Set-Location $WEB_ROOT

$ts = [string](Get-Date).ToFileTimeUtc()
$devUrl = "http://localhost:5500/?_ts=$ts"
Start-Process $devUrl

Write-Host ""
Write-Host ("logs live in ${LOG_DIR}:")
Write-Host ("  actions_server.log  -> ${actionsLog}")
Write-Host ("  rasa_core.log       -> ${coreLog}")
Write-Host ""
Write-Host "Ctrl+C to stop the frontend; background jobs will be cleaned up."
Write-Host ""

try {
    & $VENV_PYTHON -m http.server 5500
}
finally {
    Write-Host ""
    Write-Host "stopping background jobs..."

    Stop-Job $actionsJob.Id -ErrorAction SilentlyContinue
    Stop-Job $coreJob.Id    -ErrorAction SilentlyContinue

    Receive-Job $actionsJob.Id | Out-Null
    Receive-Job $coreJob.Id    | Out-Null

    Remove-Job $actionsJob.Id,$coreJob.Id -ErrorAction SilentlyContinue

    Write-Host "done."
    Write-Host ("Logs staan in ${LOG_DIR}")
}
