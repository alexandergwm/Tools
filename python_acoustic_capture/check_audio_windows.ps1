$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if ($env:ACOUSTIC_CAPTURE_ENABLE_ASIO -ne "0") {
    $env:SD_ENABLE_ASIO = "1"
}

$selectionFile = Join-Path $PSScriptRoot ".python-selection.json"
if (-not (Test-Path -LiteralPath $selectionFile)) {
    throw "Run start_gui_windows.bat once and select a Python interpreter first."
}
$selection = Get-Content -LiteralPath $selectionFile -Raw -Encoding UTF8 | ConvertFrom-Json
$runtimePython = [string]$selection.base_python
if ($selection.mode -eq "local") {
    $runtimePython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $runtimePython -PathType Leaf)) {
    throw "The configured Python interpreter does not exist. Run choose_python_windows.bat."
}

Write-Host "============================================================"
Write-Host "Audio devices visible to PortAudio"
Write-Host "============================================================"
& $runtimePython -m acoustic_capture devices
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "============================================================"
Write-Host "Validate devices, 48 kHz, and channel counts"
Write-Host "============================================================"
& $runtimePython -m acoustic_capture hardware-check configs\rme_ucx.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "============================================================"
Write-Host "Open a one-second silent full-duplex stream"
Write-Host "============================================================"
& $runtimePython -m acoustic_capture check-duplex configs\rme_ucx.yaml --duration 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$answer = Read-Host "Run a five-second microphone-only recording test? [y/N]"
if ($answer -match "^[Yy]") {
    & $runtimePython -m acoustic_capture check-input configs\rme_ucx.yaml --duration 5
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "Microphone recording test completed. Results are in the runs directory."
}

exit 0
