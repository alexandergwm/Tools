$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ReleaseRoot = Join-Path $ProjectRoot "portable_release"
$BuildRoot = Join-Path $ReleaseRoot "build"
$DistRoot = Join-Path $ReleaseRoot "dist"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Virtual environment not found: $PythonExe"
}

$PyInstallerPackage = Join-Path $ProjectRoot ".venv\Lib\site-packages\PyInstaller"
if (-not (Test-Path -LiteralPath $PyInstallerPackage)) {
    & $PythonExe -m pip install "PyInstaller>=6.0,<7"
}

& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --workpath $BuildRoot `
    --distpath $DistRoot `
    (Join-Path $ProjectRoot "AcousticCapture.spec")

& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --workpath $BuildRoot `
    --distpath $DistRoot `
    (Join-Path $ProjectRoot "AcousticCaptureOneFile.spec")

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
