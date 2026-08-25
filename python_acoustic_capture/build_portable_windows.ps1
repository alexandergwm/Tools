param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ReleaseRoot = Join-Path $ProjectRoot "portable_release"
$BuildRoot = Join-Path $ReleaseRoot "build"
$DistRoot = Join-Path $ReleaseRoot "dist"

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    if (Test-Path -LiteralPath $ProjectPython) {
        $PythonExe = $ProjectPython
    } else {
        $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $PythonCommand) {
            throw "Python not found. Create .venv or pass -PythonExe C:\path\to\python.exe"
        }
        $PythonExe = $PythonCommand.Source
    }
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python interpreter not found: $PythonExe"
}

& $PythonExe -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed for $PythonExe"
}

& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --workpath $BuildRoot `
    --distpath $DistRoot `
    (Join-Path $ProjectRoot "AcousticCapture.spec")
if ($LASTEXITCODE -ne 0) {
    throw "Folder Portable build failed with exit code $LASTEXITCODE"
}

& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --workpath $BuildRoot `
    --distpath $DistRoot `
    (Join-Path $ProjectRoot "AcousticCaptureOneFile.spec")
if ($LASTEXITCODE -ne 0) {
    throw "Single-file Portable build failed with exit code $LASTEXITCODE"
}

$FolderPackage = Join-Path $DistRoot "AcousticCapturePortable"
$ReadmeSource = Join-Path $ProjectRoot "portable_assets\README_PORTABLE.txt"
Copy-Item -LiteralPath $ReadmeSource -Destination $FolderPackage -Force

$ZipPath = Join-Path $ReleaseRoot "AcousticCapturePortable-folder.zip"
if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
Compress-Archive -LiteralPath $FolderPackage -DestinationPath $ZipPath -CompressionLevel Optimal

$SingleExe = Join-Path $DistRoot "AcousticCapturePortable.exe"
$ChecksumPath = Join-Path $ReleaseRoot "SHA256SUMS.txt"
$ChecksumLines = @(
    "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $SingleExe).Hash, "dist/AcousticCapturePortable.exe"
    "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $ZipPath).Hash, "AcousticCapturePortable-folder.zip"
)
[System.IO.File]::WriteAllLines(
    $ChecksumPath,
    $ChecksumLines,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host "Portable folder: $FolderPackage"
Write-Host "Portable ZIP:    $ZipPath"
Write-Host "Single EXE:      $SingleExe"
Write-Host "SHA256 file:     $ChecksumPath"
