param([switch]$Docker, [switch]$NoBrowser, [int]$Port = 7860, [int]$Threads = 4)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Read-LocalJson([string]$route) {
    $request = [System.Net.HttpWebRequest]::Create("http://127.0.0.1:$Port$route")
    $request.Proxy = $null
    $request.Timeout = 2000
    $response = $request.GetResponse()
    try { $reader = New-Object System.IO.StreamReader($response.GetResponseStream()); return ($reader.ReadToEnd() | ConvertFrom-Json) }
    finally { $response.Dispose() }
}
$socket = New-Object System.Net.Sockets.TcpClient
$portInUse = $false
try { $connected = $socket.ConnectAsync('127.0.0.1', $Port); $portInUse = $connected.Wait(1000) -and $socket.Connected } catch { }
finally { $socket.Dispose() }
if ($portInUse) {
    try { $existing = Read-LocalJson '/health'; $properties = Read-LocalJson '/props' }
    catch { throw "Port $Port is occupied. Run start.ps1 -Port 7861 to use another port." }
    if ($existing.status -ne 'ok' -or $properties.model_alias -ne 'guidelinekg') { throw "Port $Port is occupied. Run start.ps1 -Port 7861 to use another port." }
    Write-Host "Guideline Q&A is already running at http://localhost:$Port"
    if (-not $NoBrowser) { Start-Process "http://localhost:$Port" }
    exit 0
}
if ($Docker) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Install and start Docker Desktop, then try again.' }
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'runtime/docker-images-amd64.tar')) {
        & docker image inspect 'ghcr.io/ggml-org/llama.cpp:server-b11429' *> $null
        if ($LASTEXITCODE -ne 0) { & docker load --input 'runtime/docker-images-amd64.tar'; if ($LASTEXITCODE -ne 0) { throw 'Docker image import failed.' } }
    }
    $env:PORT = "$Port"
    $env:CPU_THREADS = "$Threads"
    & docker compose up -d --wait --wait-timeout 180
    if ($LASTEXITCODE -ne 0) { throw 'Docker startup failed. Start Docker Desktop and inspect docker compose logs.' }
} else {
    foreach ($dependency in @('VCRUNTIME140.dll', 'VCRUNTIME140_1.dll', 'MSVCP140.dll')) {
        if (-not (Test-Path -LiteralPath (Join-Path $env:WINDIR "System32/$dependency"))) {
            throw 'Microsoft Visual C++ runtime is needed. Run runtime/windows/VC_redist.x64.exe once, then double-click start.cmd again.'
        }
    }
    $runtime = Join-Path $PSScriptRoot 'runtime/windows/llama-server.exe'
    $modelName = if ($env:MODEL_FILE) { $env:MODEL_FILE } else { 'Llama-GuidelineKG-1B-Q8_0.gguf' }
    $model = Join-Path $PSScriptRoot "models/$modelName"
    $web = Join-Path $PSScriptRoot 'web'
    if (-not (Test-Path -LiteralPath $runtime)) { throw 'Bundled Windows runtime is missing. Use start.ps1 -Docker instead.' }
    if (-not (Test-Path -LiteralPath $model)) { throw 'The bundled model file is missing. Extract the complete university folder.' }
    $logDir = Join-Path $PSScriptRoot 'logs'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $arguments = @('--model', ('"{0}"' -f $model), '--path', ('"{0}"' -f $web),
        '--alias', 'guidelinekg', '--host', '127.0.0.1', '--port', "$Port", '--ctx-size', '4096',
        '--parallel', '1', '--threads', "$Threads", '--threads-batch', "$Threads", '--n-gpu-layers', '0',
        '--predict', '512', '--temp', '0', '--repeat-penalty', '1.3', '--jinja')
    $process = Start-Process -FilePath $runtime -ArgumentList $arguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logDir 'model.stdout.log') -RedirectStandardError (Join-Path $logDir 'model.stderr.log')
    $process.Id | Set-Content -LiteralPath (Join-Path $logDir "model-$Port.pid")
    Write-Host 'Loading the bundled model on CPU...'
    $ready = $false
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        $process.Refresh()
        if ($process.HasExited) { throw "The model could not start. Read logs/model.stderr.log. Choose another port if $Port is occupied." }
        try { $health = Read-LocalJson '/health'; if ($health.status -eq 'ok') { $ready = $true; break } } catch { }
        Start-Sleep -Seconds 1
    }
    if (-not $ready) { Stop-Process -Id $process.Id; throw 'The model did not become ready within 90 seconds. Read logs/model.stderr.log.' }
}
Write-Host "Guideline Q&A is ready: http://localhost:$Port"
if (-not $NoBrowser) { Start-Process "http://localhost:$Port" }
