# A-14.1 CPU e2e acceptance: q4e-server (prefetch ON by default, no --q4e-prefetch flag) + real GGUF
# Run from repo root (cwd inherited). PowerShell 5.1 compatible. ASCII only, no CJK literals.
# Safety: kill ONLY the q4e-server PID started by this script; NEVER touch the harness llama-server.
$ErrorActionPreference = 'Continue'

$ev = 'evidence_a141'
if (-not (Test-Path $ev)) { New-Item -ItemType Directory -Path $ev | Out-Null }
$errLog = Join-Path $ev 'server.err.log'
$outLog = Join-Path $ev 'server.out.log'
$svr = $null

# --- self-contained request bodies (created here; do not depend on legacy body files) ---
$bodyNoAuth = '{"messages":[{"role":"user","content":"ping"}],"max_tokens":1,"temperature":0,"stream":false}'
$bodyG3 = '{"messages":[{"role":"user","content":"What is a cache? Explain in one sentence."}],"max_tokens":64,"temperature":0,"stream":false}'
$bodyG4 = '{"messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":8,"temperature":0,"stream":true}'
Set-Content -Path (Join-Path $ev 'body_nounauth.json') -Value $bodyNoAuth -Encoding ASCII
Set-Content -Path (Join-Path $ev 'body_g3.json') -Value $bodyG3 -Encoding ASCII
Set-Content -Path (Join-Path $ev 'body_g4.json') -Value $bodyG4 -Encoding ASCII
Write-Host "setup: request bodies written to $ev"

function Fail([string]$gate, [string]$msg) {
  Write-Host ("{0} FAIL: {1}" -f $gate, $msg)
  foreach ($lf in @($errLog, $outLog)) {
    if (Test-Path $lf) {
      Write-Host ("--- tail(30) {0} ---" -f $lf)
      Get-Content $lf -Tail 30
    }
  }
  if ($svr) { Stop-Process -Id $svr.Id -Force -ErrorAction SilentlyContinue }
  exit 1
}

# G0: no residual q4e-server process; port 8123 not listening
$left0 = Get-Process -Name 'q4e-server' -ErrorAction SilentlyContinue
if ($left0) { Fail 'G0' ("residual q4e-server pid(s): " + (($left0.Id) -join ',')) }
$listen0 = Get-NetTCPConnection -LocalPort 8123 -State Listen -ErrorAction SilentlyContinue
if ($listen0) { Fail 'G0' ("port 8123 already listening, pid(s): " + (($listen0.OwningProcess) -join ',')) }
Write-Host 'G0 PASS: no q4e-server process; port 8123 not listening (method: Get-NetTCPConnection -LocalPort 8123 -State Listen)'

# START: launch q4e-server (prefetch is ON by default in this build; no --q4e-prefetch flag)
try {
  $svr = Start-Process -FilePath 'engine\build-msvc\bin\q4e-server.exe' -ArgumentList @(
    '--model','D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
    '--host','127.0.0.1','--port','8123','--api-key','a141key',
    '--ctx-size','2048','--parallel','1'
  ) -RedirectStandardOutput $outLog -RedirectStandardError $errLog -NoNewWindow -PassThru
} catch {
  Write-Host ("START FAIL: " + $_.Exception.Message)
  exit 1
}
if (-not $svr -or $svr.Id -le 0) { Write-Host 'START FAIL: Start-Process returned no pid'; exit 1 }
Write-Host ("START: q4e-server launched, pid={0}" -f $svr.Id)

# G1: /health 200 within 480s (poll every 10s); fail if the process exits first
$g1sw = [System.Diagnostics.Stopwatch]::StartNew()
$deadline = (Get-Date).AddSeconds(480)
$g1ok = $false
$lastCode = ''
while ((Get-Date) -lt $deadline) {
  $c = curl.exe -s -o NUL -w "%{http_code}" --max-time 10 http://127.0.0.1:8123/health 2>$null
  $lastCode = ("$c").Trim()
  if ($lastCode -eq '200') { $g1ok = $true; break }
  if ($svr.HasExited) { break }
  Start-Sleep -Seconds 10
}
$g1sec = [math]::Round($g1sw.Elapsed.TotalSeconds, 0)
if ($g1ok) { Write-Host ("G1 PASS: /health 200 after {0}s (poll 10s, deadline 480s)" -f $g1sec) }
else { Fail 'G1' ("health never returned 200 (last code={0}, process exited={1}) after {2}s" -f $lastCode, $svr.HasExited, $g1sec) }

# G2: POST /v1/chat/completions WITHOUT auth -> expect 401
$g2 = curl.exe -s -o NUL -w "%{http_code}" --max-time 30 -X POST http://127.0.0.1:8123/v1/chat/completions -H "Content-Type: application/json" -d ('@' + (Join-Path $ev 'body_nounauth.json')) 2>$null
$g2 = ("$g2").Trim()
if ($g2 -eq '401') { Write-Host 'G2 PASS: no-auth POST -> 401' }
else { Fail 'G2' ("expected 401, got $g2") }

# G3: 64-token non-streaming completion -> expect 200 and completion_tokens >= 1
$g3w = curl.exe -s -o (Join-Path $ev 'gen64.json') -w "%{http_code} time=%{time_total}" --max-time 600 -X POST http://127.0.0.1:8123/v1/chat/completions -H "Authorization: Bearer a141key" -H "Content-Type: application/json" -d ('@' + (Join-Path $ev 'body_g3.json')) 2>$null
$g3w = ("$g3w").Trim()
$genRaw = ''
if (Test-Path (Join-Path $ev 'gen64.json')) { $genRaw = Get-Content (Join-Path $ev 'gen64.json') -Raw }
$ct = 0
$m = [regex]::Match($genRaw, '"completion_tokens":\s*(\d+)')
if ($m.Success) { $ct = [int]$m.Groups[1].Value }
if ($g3w -match '^200' -and $ct -ge 1) {
  Write-Host ("G3 PASS: {0} completion_tokens={1}" -f $g3w, $ct)
} else {
  Fail 'G3' ("expected 200 + completion_tokens>=1, got: {0} completion_tokens={1} body={2}" -f $g3w, $ct, $genRaw)
}

# G4: SSE stream -> expect >= 2 data: lines and one data: [DONE]
$g4raw = curl.exe -s -N --max-time 300 -X POST http://127.0.0.1:8123/v1/chat/completions -H "Authorization: Bearer a141key" -H "Content-Type: application/json" -d ('@' + (Join-Path $ev 'body_g4.json')) 2>$null
$g4text = (@($g4raw) -join "`n")
Set-Content -Path (Join-Path $ev 'sse.txt') -Value $g4text
$dataLines = @($g4text -split "`r?`n" | Where-Object { $_ -match '^\s*data:' })
$hasDone = [bool]($g4text -match 'data:\s*\[DONE\]')
if ($dataLines.Count -ge 2 -and $hasDone) {
  Write-Host ("G4 PASS: SSE {0} data lines + [DONE] (saved sse.txt)" -f $dataLines.Count)
} else {
  Fail 'G4' ("SSE check failed: {0} data lines, done={1}, raw={2}" -f $dataLines.Count, $hasDone, $g4text)
}

# G5': GET /q4e/stats with auth -> save raw stats.json; gate on prefetch fields
$g5code = curl.exe -s -o (Join-Path $ev 'stats.json') -w "%{http_code}" --max-time 30 http://127.0.0.1:8123/q4e/stats -H "Authorization: Bearer a141key" 2>$null
$g5code = ("$g5code").Trim()
$statsRaw = ''
if (Test-Path (Join-Path $ev 'stats.json')) { $statsRaw = Get-Content (Join-Path $ev 'stats.json') -Raw }
if ($g5code -ne '200') { Fail 'G5' ("GET /q4e/stats -> {0}, body={1}" -f $g5code, $statsRaw) }

$stats = $null
try { $stats = $statsRaw | ConvertFrom-Json } catch { Fail 'G5' ("stats.json is not JSON: " + $_.Exception.Message) }

$failures = @()
if ($stats.q4e_prefetch -ne 1) { $failures += ("q4e_prefetch={0} (want 1)" -f $stats.q4e_prefetch) }
if ($stats.touches -le 0) { $failures += ("touches={0} (want >0)" -f $stats.touches) }
if ($stats.layers_touched -ne 48) { $failures += ("layers_touched={0} (want 48)" -f $stats.layers_touched) }
if ($stats.prefetch_io_issued -le 0) { $failures += ("prefetch_io_issued={0} (want >0)" -f $stats.prefetch_io_issued) }
if ($stats.prefetch_overlap_hits -le 0) { $failures += ("prefetch_overlap_hits={0} (want >0)" -f $stats.prefetch_overlap_hits) }

if ($failures.Count -eq 0) {
  Write-Host 'G5 PASS: gates q4e_prefetch==1, touches>0, layers_touched==48, prefetch_io_issued>0, prefetch_overlap_hits>0'
} else {
  Write-Host ("stats.json: {0}" -f $statsRaw)
  Fail 'G5' (($failures) -join ' | ')
}

Write-Host ('G5 ungated: prefetch_io_bytes={0} prefetch_io_completed={1} prefetch_io_dropped={2} prefetch_events_dropped={3} prefetch_overlap_ratio={4} hook_skips_weights_device={5} hits_vram={6} hits_ram={7} hits_ssd={8}' -f `
  $stats.prefetch_io_bytes, $stats.prefetch_io_completed, $stats.prefetch_io_dropped, $stats.prefetch_events_dropped, $stats.prefetch_overlap_ratio, $stats.hook_skips_weights_device, $stats.hits_vram, $stats.hits_ram, $stats.hits_ssd)

$warmLine = ''
foreach ($lf in @($errLog, $outLog)) {
  if (Test-Path $lf) {
    $hit = Select-String -Path $lf -Pattern 'q4e: live prefetch backend active' | Select-Object -First 1
    if ($hit) { $warmLine = $hit.Line.Trim(); break }
  }
}
if ($warmLine) { Write-Host ("G5 warm-backend log line: {0}" -f $warmLine) }
else { Write-Host 'G5 NOTE: "q4e: live prefetch backend active" line not found in server logs (not gated)' }

# G6: kill ONLY this script's pid; wait up to 180s (poll 2s) until gone; assert no q4e-server remains
Write-Host ("G6: sending SIGKILL to pid {0}" -f $svr.Id)
Stop-Process -Id $svr.Id -Force -ErrorAction SilentlyContinue
$g6sw = [System.Diagnostics.Stopwatch]::StartNew()
$gone = $false
while ($g6sw.Elapsed -lt [TimeSpan]::FromSeconds(180)) {
  $p = Get-Process -Id $svr.Id -ErrorAction SilentlyContinue
  if (-not $p) { $gone = $true; break }
  Start-Sleep -Seconds 2
}
$anyQ4e = Get-Process -Name 'q4e-server' -ErrorAction SilentlyContinue
if ($gone -and -not $anyQ4e) {
  Write-Host ("G6 PASS: pid {0} gone after {1}s; no residual q4e-server" -f $svr.Id, [math]::Round($g6sw.Elapsed.TotalSeconds, 0))
} else {
  $resId = ''
  if ($anyQ4e) { $resId = (($anyQ4e.Id) -join ',') }
  Fail 'G6' ("q4e-server residual after SIGKILL wait (pids: {0})" -f $resId)
}

Write-Host 'A141_E2E_ALL_PASS'
exit 0
