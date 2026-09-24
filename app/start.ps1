# One-command start for Windows: builds the frontend and creates the Python environment if needed,
# then serves the API and the UI together at http://localhost:8000
$root = $PSScriptRoot

if (-not (Test-Path "$root\frontend\dist")) {
    Push-Location "$root\frontend"
    npm install
    npm run build
    Pop-Location
}

Push-Location "$root\backend"
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    .\.venv\Scripts\python -m pip install -r requirements.txt
}
Write-Host "AeroNet running at http://localhost:8000  (Ctrl+C to stop)"
.\.venv\Scripts\python -m uvicorn aeronet.main:app --port 8000
Pop-Location
