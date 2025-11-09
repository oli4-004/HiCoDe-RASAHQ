# -------------------------------------------------
# CampusCompass\run-dev.ps1
# Start:
# - Action server (5055)
# - Rasa server (5005)
# - Frontend static server (5500)
# Waits until Rasa is up before opening the frontend.
# Keeps the window open on failure so you can see errors.
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

Write-Host "============================================="
Write-Host " CampusCompass - DEV"
Write-Host " BOT_ROOT     = $BOT_ROOT"
Write-Host " PROJECT_ROOT = $PROJECT_ROOT"
Write-Host " VENV_PYTHON  = $VENV_PYTHON"
Write-Host " VENV_RASA    = $VENV_RASA"
Write-Host " LOG_DIR      = $LOG_DIR"
Write-Host "============================================="

if (!(Test-Path $VENV_PYTHON)) {
    Write-Host "ERROR: No python venv found at:"
    Write-Host "  $VENV_PYTHON"
    Write-Host "Create venv and pip install -r requirements.txt first."
    Read-Host "Press Enter to exit"
    exit 1
}

if (!(Test-Path $VENV_RASA)) {
    Write-Host "ERROR: No rasa.exe found at:"
    Write-Host "  $VENV_RASA"
    Write-Host "Activate the venv and install Rasa in it."
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "venv & rasa OK"

function Kill-Port {
    param([int]$port)

    $conns = netstat -ano | Select-String (":$port") 2>$null
    foreach ($line in $conns) {
        $parts = $line.ToString().Trim() -split "\s+"
        if ($parts.Length -ge 5) {
            $pid_train = $parts[-1]
            try {
                taskkill /PID $pid_train /F 2>$null | Out-Null
                Write-Host ("killed PID {0} on port {1}" -f $pid_train, $port)
            } catch {
                # ignore
            }
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

# Kill any old processes
Write-Host "Cleaning up old processes on 5005, 5055, 5500 ..."
Kill-Port 5005
Kill-Port 5055
Kill-Port 5500

# 1) Start ACTION SERVER
Write-Host "Starting action server on :5055 ..."
$actionsJob = Start-Job -ScriptBlock {
    param($PROJECT_ROOT, $VENV_RASA, $actionsLog)
    Set-Location $PROJECT_ROOT

    & $VENV_RASA run actions `
        --actions CampusCompass.app.actions.actions `
        --port 5055 `
        *> $actionsLog

} -ArgumentList $PROJECT_ROOT, $VENV_RASA, $actionsLog

Write-Host ("Action server job ID = {0}" -f $actionsJob.Id)
Start-Sleep -Seconds 2

# 2) Start RASA SERVER
Write-Host "Starting Rasa server on :5005 ..."
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

Write-Host ("Rasa core job ID     = {0}" -f $coreJob.Id)

# 3) Wait until Rasa is ready
Write-Host "Waiting for Rasa to become ready..."
$ready = Wait-For-Rasa
if (-not $ready) {
    Write-Host "WARNING: Rasa not ready. Check:"
    Write-Host "  $coreLog"
    Write-Host "Continuing anyway so you can inspect logs."
}

# 4) Start frontend
Write-Host "Starting frontend at http://localhost:5500 ..."
$WEB_ROOT = Join-Path $BOT_ROOT "web"
if (!(Test-Path $WEB_ROOT)) {
    Write-Host "ERROR: Web root not found at:"
    Write-Host "  $WEB_ROOT"
    Write-Host "Make sure your 'web' folder exists next to this script."
    # stop jobs so we don't leak them
    Stop-Job $actionsJob.Id -ErrorAction SilentlyContinue
    Stop-Job $coreJob.Id    -ErrorAction SilentlyContinue
    Receive-Job $actionsJob.Id -ErrorAction SilentlyContinue | Out-Null
    Receive-Job $coreJob.Id    -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $actionsJob.Id,$coreJob.Id -ErrorAction SilentlyContinue
    Read-Host "Press Enter to exit"
    exit 1
}

Set-Location $WEB_ROOT

$ts = [string](Get-Date).ToFileTimeUtc()
$devUrl = "http://localhost:5500/?_ts=$ts"
Start-Process $devUrl

Write-Host ""
Write-Host "Logs:"
Write-Host "  Action server: $actionsLog"
Write-Host "  Rasa core:     $coreLog"
Write-Host ""
Write-Host "Frontend running on http://localhost:5500"
Write-Host "Press Ctrl+C in this window to stop the frontend and servers."
Write-Host ""

try {
    # Block here so the window stays open while http.server runs.
    & $VENV_PYTHON -m http.server 5500
}
catch {
    Write-Host "ERROR: Failed to start Python http.server on port 5500"
    Write-Host $_.Exception.Message
    Read-Host "Press Enter to exit"
}
finally {
    Write-Host ""
    Write-Host "Stopping background jobs..."

    Stop-Job $actionsJob.Id -ErrorAction SilentlyContinue
    Stop-Job $coreJob.Id    -ErrorAction SilentlyContinue

    Receive-Job $actionsJob.Id -ErrorAction SilentlyContinue | Out-Null
    Receive-Job $coreJob.Id    -ErrorAction SilentlyContinue | Out-Null

    Remove-Job $actionsJob.Id,$coreJob.Id -ErrorAction SilentlyContinue

    Write-Host "Done."
    Write-Host ("Logs are in $LOG_DIR")
    # Als je dubbelklikt wil je de output nog zien:
    Read-Host "Press Enter to exit"
}
