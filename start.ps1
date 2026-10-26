param([switch]$Demo, [string]$Camera = '1', [double]$Dwell = 1.1, [double]$HoldTolerance = 0.06, [double]$HoldIou = 0.35, [int]$MinRoiPixels = 4, [int]$CameraWidth = 1920, [int]$CameraHeight = 1080)
$ErrorActionPreference = 'Stop'
$pythonExe = 'C:\ProgramData\Anaconda3\envs\mediapipe\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Cannot find the mediapipe Python environment. Update pythonExe in start.ps1.' }
$launchArgs = @('-s', (Join-Path $PSScriptRoot 'main.py'), '--camera', $Camera)
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$launchArgs += @('--dwell', $Dwell.ToString($culture), '--hold-tolerance', $HoldTolerance.ToString($culture), '--hold-iou', $HoldIou.ToString($culture))
$launchArgs += @('--min-roi-pixels', "$MinRoiPixels")
$launchArgs += @('--camera-width', "$CameraWidth", '--camera-height', "$CameraHeight")
if ($Demo) { $launchArgs += '--demo' }
& $pythonExe @launchArgs
