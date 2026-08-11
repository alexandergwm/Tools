param([switch]$CheckOnly)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if ($env:ACOUSTIC_CAPTURE_ENABLE_ASIO -ne "0") {
    $env:SD_ENABLE_ASIO = "1"
}

$selectionFile = Join-Path $PSScriptRoot ".python-selection.json"
$selector = Join-Path $PSScriptRoot "select_python_windows.ps1"
if (-not (Test-Path -LiteralPath $selectionFile)) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $selector
    if ($LASTEXITCODE -ne 0) {
        throw "Python interpreter selection was cancelled or failed."
    }
}

$selection = Get-Content -LiteralPath $selectionFile -Raw -Encoding UTF8 | ConvertFrom-Json
$basePython = [string]$selection.base_python
$runtimePython = $basePython
$needInstall = $false

if ($selection.mode -eq "local") {
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Write-Host "[Acoustic Capture] Creating the local Python environment..."
        & $basePython -m venv (Join-Path $PSScriptRoot ".venv")
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create the local Python environment."
        }
        $needInstall = $true
    }
    $runtimePython = $venvPython
}

if (-not (Test-Path -LiteralPath $runtimePython -PathType Leaf)) {
    throw "The selected Python interpreter no longer exists. Run choose_python_windows.bat and select it again."
}

& $runtimePython -c "import struct,sys; raise SystemExit(0 if sys.version_info >= (3,10) and struct.calcsize('P') == 8 else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Acoustic Capture requires a 64-bit Python 3.10 or newer interpreter."
}

& $runtimePython -c "from importlib.metadata import version; import acoustic_capture,numpy,scipy,soundfile,sounddevice,matplotlib,yaml,xlsxwriter; raise SystemExit(0 if tuple(map(int,version('sounddevice').split('.')[:3])) >= (0,5,1) else 1)" 2>$null
if ($LASTEXITCODE -ne 0) {
    $needInstall = $true
}

if ($needInstall) {
    Write-Host "[Acoustic Capture] Installing the application and dependencies..."
    & $runtimePython -m pip install -e .
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed. Check the network connection and Python environment."
    }
}

if ($CheckOnly) {
    Write-Host "[Acoustic Capture] Python environment check passed: $runtimePython"
    exit 0
}

$demoTarget = Join-Path $PSScriptRoot "audio\targets\demo_target_001.wav"
if (-not (Test-Path -LiteralPath $demoTarget)) {
    Write-Host "[Acoustic Capture] Creating workflow test audio..."
    & $runtimePython -m acoustic_capture demo-audio --output-dir audio
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create workflow test audio."
    }
}

Write-Host "[Acoustic Capture] Opening the Windows audio capture interface..."
& $runtimePython -m acoustic_capture gui configs\rme_ucx.yaml
exit $LASTEXITCODE
