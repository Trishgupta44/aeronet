<#
  Regenerate the saved results (the static demo's bundle, and the offline fallback) for a PUBLIC deployment.

    .\deploy\prepare-public.ps1              include mode (default): NNfoil's reference data is part of the bundle (the author allowed it;
                                             see docs\DEPLOY.md, "Licence"). Needed only when the results or the app's answers change.
    .\deploy\prepare-public.ps1 -Withhold    strict mode: builds a stripped copy of the data (deploy\data\public, git-ignored), starts a backend
                                             that withholds the FVM data and uses only that copy, writes the bundle from it and checks that it
                                             holds no FVM data. Do NOT commit the strict bundle over the normal one (git checkout it after building).

  Run from the repository root (Windows PowerShell). Needs the backend venv (app\backend\.venv) and Node.
  Then build the static site (cd app\frontend; npm run build:static, or build:static:strict) or the image (docs\DEPLOY.md).
#>
param(
  [switch]$Withhold,
  [int]$Port = 8099
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root 'app\backend\.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "Backend venv not found at $py (see app\README.md)" }

if ($Withhold) {
  $data = Join-Path $root 'deploy\data\public'
  & $py (Join-Path $root 'deploy\make_data.py') --out $data
  if ($LASTEXITCODE) { throw 'make_data.py failed' }
  $env:AERONET_WITHHOLD_FVM_DATA = '1'
  $env:AERONET_DATA_DIR = $data      # proves the stripped copy is enough: the full data is not visible to this backend
} else {
  Remove-Item Env:AERONET_WITHHOLD_FVM_DATA -ErrorAction SilentlyContinue
  Remove-Item Env:AERONET_DATA_DIR -ErrorAction SilentlyContinue
}
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
    if ($Withhold) { node scripts/check-public-bundle.mjs --withhold } else { node scripts/check-public-bundle.mjs }
    if ($LASTEXITCODE) { throw 'the bundle does not fit the chosen mode' }
  } finally { Pop-Location }
} finally {
  if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
  Remove-Item Env:AERONET_WITHHOLD_FVM_DATA, Env:AERONET_DATA_DIR -ErrorAction SilentlyContinue
}
if ($Withhold) {
  Write-Host 'Ready (strict): cd app/frontend; npm run build:static:strict   or   docker build -f deploy/Dockerfile --build-arg WITHHOLD_FVM=1 -t aeronet-strict .'
} else {
  Write-Host 'Ready: cd app/frontend; npm run build:static   or   docker build -f deploy/Dockerfile -t aeronet .'
}
