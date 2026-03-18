$ErrorActionPreference = "Stop"

param(
    [int]$Port = 8000
)

function Get-NgrokPublicUrl {
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 2
        $json = $resp.Content | ConvertFrom-Json
        $httpsTunnel = $json.tunnels | Where-Object { $_.proto -eq "https" } | Select-Object -First 1
        if ($httpsTunnel) {
            return $httpsTunnel.public_url
        }
    } catch {
        return $null
    }
    return $null
}

function Test-LocalPort {
    param([int]$CheckPort)
    try {
        $result = Test-NetConnection -ComputerName 127.0.0.1 -Port $CheckPort -WarningAction SilentlyContinue
        return [bool]$result.TcpTestSucceeded
    } catch {
        return $false
    }
}

$repoRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($repoRoot)) {
    $repoRoot = (Get-Location).Path
}

Write-Host "[1/3] Checking backend on port $Port..."
$backendUp = Test-LocalPort -CheckPort $Port
if (-not $backendUp) {
    Write-Host "Starting Flask backend..."
    $backendCmd = "Set-Location -Path '$repoRoot'; python backend/app.py"
    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $backendCmd | Out-Null
} else {
    Write-Host "Backend already running."
}

Write-Host "[2/3] Checking ngrok tunnel..."
$publicUrl = Get-NgrokPublicUrl
if (-not $publicUrl) {
    Write-Host "Starting ngrok tunnel..."
    $ngrokCmd = "Set-Location -Path '$repoRoot'; ngrok http $Port"
    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $ngrokCmd | Out-Null
}

Write-Host "[3/3] Waiting for ngrok URL..."
$tries = 30
for ($i = 0; $i -lt $tries; $i++) {
    Start-Sleep -Seconds 1
    $publicUrl = Get-NgrokPublicUrl
    if ($publicUrl) { break }
}

if ($publicUrl) {
    Write-Host ""
    Write-Host "Public URL: $publicUrl"
    Write-Host "Health check: $publicUrl/api/health"
    Write-Host ""
    Write-Host "Keep both terminal windows open while testing on phone."
    exit 0
}

Write-Host ""
Write-Host "ngrok URL not detected yet. Check the ngrok terminal for errors."
exit 1
