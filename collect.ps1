param([string]$Operator = 'p01', [string]$Camera = '1')
$ErrorActionPreference = 'Stop'
$pythonExe = 'C:\ProgramData\Anaconda3\envs\mediapipe\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Cannot find mediapipe Python. Update pythonExe in collect.ps1.' }
& $pythonExe -s (Join-Path $PSScriptRoot 'collect_digits.py') --operator $Operator --camera $Camera
