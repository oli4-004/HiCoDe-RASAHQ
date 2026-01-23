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
# ---- Cross-platform venv detection (Windows + macOS + Linux) ----
$VENV_DIR = Join-Path $BOT_ROOT ".venv"

$ON_WINDOWS = $env:OS -eq "Windows_NT"
if (-not $ON_WINDOWS) {
    # On macOS/Linux we should be running in PowerShell 7+, so this exists:
    $osString = $PSVersionTable.OS
    $ON_MAC   = $osString -match "Darwin"
    $ON_LINUX = -not $ON_MAC
} else {
    $ON_MAC = $false
    $ON_LINUX = $false
}

if ($ON_WINDOWS) {
    $VENV_PYTHON = Join-Path (Join-Path $VENV_DIR "Scripts") "python.exe"
    $VENV_RASA   = Join-Path (Join-Path $VENV_DIR "Scripts") "rasa.exe"
} else {
    $VENV_PYTHON = Join-Path (Join-Path $VENV_DIR "bin") "python"
    if (!(Test-Path $VENV_PYTHON)) { $VENV_PYTHON = Join-Path (Join-Path $VENV_DIR "bin") "python3" }
    $VENV_RASA   = Join-Path (Join-Path $VENV_DIR "bin") "rasa"
}

# Absolute file paths
$ENDPOINTS_PATH   = Join-Path $BOT_ROOT "endpoints.yml"
$CREDENTIALS_PATH = Join-Path $BOT_ROOT "credentials.yml"
$MODELS_PATH      = Join-Path $BOT_ROOT "models"

# Logs
$LOG_DIR = Join-Path $BOT_ROOT "logs"
if (!(Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Path $LOG_DIR | Out-Null
}
# Extra logs (truncate / create)
$mapsLog = Join-Path $LOG_DIR "maps_calls.log"
$llmLog  = Join-Path $LOG_DIR "llm_debug.log"

"" | Set-Content -Path $mapsLog -Encoding utf8
"" | Set-Content -Path $llmLog  -Encoding utf8

$ROUTES_DIR = Join-Path $BOT_ROOT "routes"
if (!(Test-Path $ROUTES_DIR)) {
    New-Item -ItemType Directory -Path $ROUTES_DIR | Out-Null
} else {
    Remove-Item -Recurse -Force (Join-Path $ROUTES_DIR "*") -ErrorAction SilentlyContinue
}

$actionsLog = Join-Path $LOG_DIR "actions_server.log"
$coreLog    = Join-Path $LOG_DIR "rasa_core.log"

Write-Host "============================================="
Write-Host " CampusCompass - DEV"
Write-Host " BOT_ROOT     = $BOT_ROOT"
Write-Host " PROJECT_ROOT = $PROJECT_ROOT"
Write-Host " VENV_PYTHON  = $VENV_PYTHON"
Write-Host " VENV_RASA    = $VENV_RASA"
Write-Host " ENDPOINTS    = $ENDPOINTS_PATH"
Write-Host " CREDENTIALS  = $CREDENTIALS_PATH"
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
    Write-Host "ERROR: No rasa executable found at:"
    Write-Host "  $VENV_RASA"
    Write-Host "Activate the venv and install Rasa in it."
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "venv & rasa OK"

function Get-PidsOnPort {
    param([int]$port)

    $pids = @()

    if ($ON_WINDOWS) {
        $conns = netstat -ano 2>$null | Select-String (":$port\s")
        foreach ($line in $conns) {
            $parts = $line.ToString().Trim() -split "\s+"
            if ($parts.Length -ge 5 -and $parts[-1] -match "^\d+$") {
                $pids += [int]$parts[-1]
            }
        }
    }
    else {
        if (Get-Command lsof -ErrorAction SilentlyContinue) {
            $out = & lsof -nP -iTCP:$port -sTCP:LISTEN -t 2>$null
            foreach ($pid in $out) {
                if ($pid -match "^\d+$") { $pids += [int]$pid }
            }
        }
        elseif (Get-Command ss -ErrorAction SilentlyContinue) {
            $out = & ss -lptn "sport = :$port" 2>$null
            foreach ($line in $out) {
                foreach ($m in [regex]::Matches($line, "pid=(\d+)")) {
                    $pids += [int]$m.Groups[1].Value
                }
            }
        }
        else {
            Write-Host "WARNING: Can't auto-detect PID on port $port (need 'lsof' or 'ss')."
        }
    }

    return $pids | Sort-Object -Unique
}

function Kill-Port {
    param([int]$port)

    $pids = Get-PidsOnPort $port
    foreach ($pid in $pids) {
        try {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            Write-Host ("killed PID {0} on port {1}" -f $pid, $port)
        } catch {
            # ignore
        }
    }
}

function Wait-For-Rasa {
    param(
        [string]$Url = "http://localhost:5005/status",
        [int]$TimeoutSeconds = 60
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -Method Get -TimeoutSec 5
            if ($r.StatusCode -eq 200) {
                $json = $r.Content | ConvertFrom-Json
                if ($json.model_file) {
                    Write-Host "Rasa is up AND model loaded at $Url"
                    return $true
                }
                Write-Host "Rasa up but still loading model..."
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
        Start-Sleep -Seconds 1
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
    param($PROJECT_ROOT, $BOT_ROOT, $VENV_RASA, $actionsLog, $ENDPOINTS_PATH)

    # Werkdirectory = CampusCompass, zodat Python 'logs/...' daar schrijft
    Set-Location $BOT_ROOT

    # CampusCompass package importable vanaf project root
    $env:PYTHONPATH = $PROJECT_ROOT

    & $VENV_RASA run actions `
        --actions CampusCompass.app.actions.actions `
        --port 5055 `
        --endpoints $ENDPOINTS_PATH `
        *> $actionsLog

} -ArgumentList $PROJECT_ROOT, $BOT_ROOT, $VENV_RASA, $actionsLog, $ENDPOINTS_PATH

Write-Host ("Action server job ID = {0}" -f $actionsJob.Id)
Start-Sleep -Seconds 2

# 2) Start RASA SERVER
Write-Host "Starting Rasa server on :5005 ..."
$coreJob = Start-Job -ScriptBlock {
    param($PROJECT_ROOT, $BOT_ROOT, $VENV_RASA, $coreLog, $CREDENTIALS_PATH, $ENDPOINTS_PATH, $MODELS_PATH)

    # Werkdirectory = CampusCompass
    Set-Location $BOT_ROOT

    # Imports vanaf project root
    $env:PYTHONPATH = $PROJECT_ROOT

    & $VENV_RASA run `
        --enable-api `
        --cors "*" `
        --port 5005 `
        --credentials $CREDENTIALS_PATH `
        --endpoints $ENDPOINTS_PATH `
        --model $MODELS_PATH `
        *> $coreLog

} -ArgumentList $PROJECT_ROOT, $BOT_ROOT, $VENV_RASA, $coreLog, $CREDENTIALS_PATH, $ENDPOINTS_PATH, $MODELS_PATH

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

    Stop-Job $actionsJob.Id -ErrorAction SilentlyContinue
    Stop-Job $coreJob.Id    -ErrorAction SilentlyContinue
    Receive-Job $actionsJob.Id -ErrorAction SilentlyContinue | Out-Null
    Receive-Job $coreJob.Id    | Out-Null
    Remove-Job $actionsJob.Id,$coreJob.Id -ErrorAction SilentlyContinue

    Read-Host "Press Enter to exit"
    exit 1
}

Set-Location $WEB_ROOT

$ts = [string](Get-Date).ToFileTimeUtc()
$devUrl = "http://localhost:5500/?_ts=$ts"
try { Start-Process $devUrl } catch { Write-Host "Note: couldn't open a browser here. Open manually: $devUrl" }
Write-Host ""
Write-Host "Logs:"
Write-Host "  Action server: $actionsLog"
Write-Host "  Rasa core:     $coreLog"
Write-Host ""
Write-Host "Frontend running on http://localhost:5500"
Write-Host "Press Ctrl+C in this window to stop the frontend and servers."
Write-Host ""

try {
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
    Read-Host "Press Enter to exit"
}