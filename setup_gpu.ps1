# Reproduce the tested project-local overlay. Does not replace the original env.
$ErrorActionPreference = 'Stop'
$pythonExe = 'C:\ProgramData\Anaconda3\envs\mediapipe\python.exe'
& $pythonExe -s -m pip install --target (Join-Path $PSScriptRoot '.deps') --no-deps torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
& $pythonExe -s -m pip install --target (Join-Path $PSScriptRoot '.extras') --no-deps -r (Join-Path $PSScriptRoot 'requirements-extras.txt')
if ($LASTEXITCODE -ne 0) { throw 'Extra dependency installation failed' }
& $pythonExe -s (Join-Path $PSScriptRoot 'check_environment.py')
