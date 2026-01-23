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
$BOT_ROOT     = $PSScriptRoot                          # ...\CampusCompass
$PROJECT_ROOT = Split-Path $BOT_ROOT -Parent           # ...\HiCoDe-RASAHQ
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

# Absolute file paths (important if we change working dir)
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

Write-Host ("BOT_ROOT       = {0}" -f $BOT_ROOT)
Write-Host ("PROJECT_ROOT   = {0}" -f $PROJECT_ROOT)
Write-Host ("VENV_PYTHON    = {0}" -f $VENV_PYTHON)
Write-Host ("VENV_RASA      = {0}" -f $VENV_RASA)
Write-Host ("ENDPOINTS      = {0}" -f $ENDPOINTS_PATH)
Write-Host ("CREDENTIALS    = {0}" -f $CREDENTIALS_PATH)
Write-Host ("LOG_DIR        = {0}" -f $LOG_DIR)

if (!(Test-Path $VENV_PYTHON)) {
    Write-Host "ERROR: No venv found at $VENV_PYTHON"
    Write-Host "Create venv and pip install -r requirements.txt first."
    exit 1
}

Write-Host "venv OK"

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
                # extra check: model loaded?
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
    --out $MODELS_PATH `
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
    param($PROJECT_ROOT, $BOT_ROOT, $VENV_RASA, $actionsLog, $ENDPOINTS_PATH)

    # Werkdirectory = CampusCompass, zodat Python 'logs/...' daar neerzet
    Set-Location $BOT_ROOT

    # CampusCompass package importable vanuit project root
    $env:PYTHONPATH = $PROJECT_ROOT

    & $VENV_RASA run actions `
        --actions CampusCompass.app.actions.actions `
        --port 5055 `
        --endpoints $ENDPOINTS_PATH `
        *> $actionsLog

} -ArgumentList $PROJECT_ROOT, $BOT_ROOT, $VENV_RASA, $actionsLog, $ENDPOINTS_PATH
Write-Host ("actions job ID = {0}" -f $actionsJob.Id)

Start-Sleep -Seconds 2

# 4) Start RASA SERVER
Write-Host "starting rasa core job on :5005 ..."
$coreJob = Start-Job -ScriptBlock {
    param($PROJECT_ROOT, $BOT_ROOT, $VENV_RASA, $coreLog, $CREDENTIALS_PATH, $ENDPOINTS_PATH, $MODELS_PATH)

    # Zelfde: werkdirectory = CampusCompass
    Set-Location $BOT_ROOT

    # Maar imports blijven vanaf project root
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
try { Start-Process $devUrl } catch { Write-Host "Note: couldn't open a browser here. Open manually: $devUrl" }
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
