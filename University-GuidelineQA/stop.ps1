$ErrorActionPreference = 'Stop'
$logDirectory = Join-Path $PSScriptRoot 'logs'
foreach ($pidFile in (Get-ChildItem -LiteralPath $logDirectory -Filter 'model*.pid' -File -ErrorAction SilentlyContinue)) {
    $modelProcessId = [int](Get-Content -LiteralPath $pidFile.FullName)
    $process = Get-Process -Id $modelProcessId -ErrorAction SilentlyContinue
    $expected = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'runtime/windows/llama-server.exe'))
    if ($process -and $process.Path -eq $expected) { Stop-Process -Id $modelProcessId; Write-Host 'Guideline Q&A stopped.' }
    Remove-Item -LiteralPath $pidFile.FullName
}
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Set-Location -LiteralPath $PSScriptRoot
    $containers = & docker ps --filter 'label=com.docker.compose.project=university-guidelineqa' --quiet 2>$null
    if ($LASTEXITCODE -eq 0 -and $containers) { & docker compose down }
}
