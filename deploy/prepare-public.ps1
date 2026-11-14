<#
  Prepare everything a PUBLIC deployment needs, without NNfoil's FVM data:
    1. deploy\data\results     the small results folder (deploy\make_data.py)
    2. app\frontend\public\cache\bundle.json   saved results made from a backend that withholds the FVM data
    3. a check that the bundle really contains none

  Run from the repository root (Windows PowerShell):
    .\deploy\prepare-public.ps1 -ResultsDir ..\Aero-Net\results

  Needs: the backend venv (app\backend\.venv) and Node. Then build the image or the static site (docs\DEPLOY.md).
#>
param(
  [Parameter(Mandatory = $true)][string]$ResultsDir,
  [int]$Port = 8099
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root 'app\backend\.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "Backend venv not found at $py (see app\README.md)" }
$data = Join-Path $root 'deploy\data\results'

& $py (Join-Path $root 'deploy\make_data.py') --results $ResultsDir --out $data
if ($LASTEXITCODE) { throw 'make_data.py failed' }

$env:AERONET_WITHHOLD_FVM_DATA = '1'
$env:AERONET_RESULTS_DIR = $data
Remove-Item Env:AERONET_FVM_DIR -ErrorAction SilentlyContinue   # proves the slim data is enough: no FVM folder is given
$server = Start-Process -FilePath $py -ArgumentList '-m', 'uvicorn', 'aeronet.main:app', '--port', $Port -WorkingDirectory (Join-Path $root 'app\backend') -WindowStyle Hidden -PassThru
try {
  $ok = $false
  for ($i = 0; $i -lt 40 -and -not $ok; $i++) {
    Start-Sleep -Milliseconds 500
    try { $ok = (Invoke-RestMethod "http://localhost:$Port/api/health").status -eq 'ok' } catch { }
  }
  if (-not $ok) { throw 'the backend did not start' }
  Push-Location (Join-Path $root 'app\frontend')
  try {
    node scripts/make-cache.mjs "http://localhost:$Port"
    if ($LASTEXITCODE) { throw 'make-cache failed' }
    node scripts/check-public-bundle.mjs
    if ($LASTEXITCODE) { throw 'the bundle still contains FVM data' }
  } finally { Pop-Location }
} finally {
  if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
}
Write-Host 'Ready: build the image (docker build -f deploy/Dockerfile -t aeronet .) or the static site (cd app/frontend; npm run build:static).'
