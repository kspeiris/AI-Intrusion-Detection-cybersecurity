$ErrorActionPreference = "Stop"
$python = "C:\Users\kavin\AppData\Local\Programs\Python\Python313\python.exe"

if (-not (Test-Path $python)) {
    throw "Python not found at $python"
}

& $python "src\realtime_detector.py"
