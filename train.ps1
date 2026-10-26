param([switch]$Quick, [switch]$Temporal, [int]$Epochs = 100, [int]$AugmentCopies = 0)
$ErrorActionPreference = 'Stop'
$pythonExe = 'C:\ProgramData\Anaconda3\envs\mediapipe\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Cannot find mediapipe Python. Update pythonExe in train.ps1.' }
$launchArgs = @('-s', (Join-Path $PSScriptRoot 'train_digits.py'), '--epochs', "$Epochs")
$launchArgs += @('--augment-copies', "$AugmentCopies")
if ($Quick) { $launchArgs += @('--group-by', 'session', '--allow-partial') }
if ($Temporal) { $launchArgs += @('--split', 'temporal') }
& $pythonExe @launchArgs
