param(
    [string]$PythonPath = "",
    [ValidateSet("local", "direct")]
    [string]$Mode = "local",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$selectionFile = Join-Path $PSScriptRoot ".python-selection.json"

if ((Test-Path -LiteralPath $selectionFile) -and -not $Force) {
    exit 0
}

Add-Type -AssemblyName System.Windows.Forms

function Test-PythonInterpreter([string]$Candidate) {
    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        return $false
    }
    & $Candidate -c "import struct,sys; raise SystemExit(0 if sys.version_info >= (3,10) and struct.calcsize('P') == 8 else 1)" 2>$null
    return $LASTEXITCODE -eq 0
}

if (-not $PythonPath) {
    [System.Windows.Forms.MessageBox]::Show(
        "Select a 64-bit Python 3.10 or newer interpreter. You may select python.exe from a normal installation, Conda environment, or existing virtual environment.",
        "Acoustic Capture - Python Setup",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Information
    ) | Out-Null

    while ($true) {
        $dialog = New-Object System.Windows.Forms.OpenFileDialog
        $dialog.Title = "Select Python interpreter"
        $dialog.Filter = "Python interpreter (python.exe)|python.exe|Executable files (*.exe)|*.exe"
        $dialog.CheckFileExists = $true
        $dialog.Multiselect = $false
        $pathCommand = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($pathCommand) {
            $dialog.InitialDirectory = Split-Path -Parent $pathCommand.Source
        }
        if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
            exit 2
        }
        $PythonPath = $dialog.FileName
        if (Test-PythonInterpreter $PythonPath) {
            break
        }
        [System.Windows.Forms.MessageBox]::Show(
            "The selected file is not a working 64-bit Python 3.10 or newer interpreter.",
            "Invalid Python interpreter",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
        $PythonPath = ""
    }

    $answer = [System.Windows.Forms.MessageBox]::Show(
        "Choose how Acoustic Capture should use this interpreter.`n`nYes: create/use this project's .venv (recommended).`nNo: use the selected environment directly.`nCancel: abort setup.",
        "Choose Python environment mode",
        [System.Windows.Forms.MessageBoxButtons]::YesNoCancel,
        [System.Windows.Forms.MessageBoxIcon]::Question
    )
    if ($answer -eq [System.Windows.Forms.DialogResult]::Cancel) {
        exit 2
    }
    $Mode = if ($answer -eq [System.Windows.Forms.DialogResult]::Yes) { "local" } else { "direct" }
}

$PythonPath = [System.IO.Path]::GetFullPath($PythonPath)
if (-not (Test-PythonInterpreter $PythonPath)) {
    throw "The selected interpreter must be a working 64-bit Python 3.10 or newer installation: $PythonPath"
}

@{
    base_python = $PythonPath
    mode = $Mode
} | ConvertTo-Json | Set-Content -LiteralPath $selectionFile -Encoding UTF8

exit 0
