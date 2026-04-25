$ErrorActionPreference = "Stop"
$python = "C:\Users\kavin\AppData\Local\Programs\Python\Python313\python.exe"

if (-not (Test-Path $python)) {
    throw "Python not found at $python"
}

& $python -m streamlit run "src\alert_dashboard.py"
