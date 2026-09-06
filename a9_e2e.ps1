# A-9 CPU e2e smoke (real GGUF, all layers host-resident -> hook touch path on 48 layers)
# NOTE: run from repo root (cwd inherited); no CJK literals (PS5.1 cp950 default)
# NOTE: match only q4e-server - the harness model server is a llama-server process, never touch it
$ErrorActionPreference = 'Continue'

# G0: no residual q4e-server, port 8123 free
$proc = Get-Process -Name 'q4e-server' -ErrorAction SilentlyContinue
if ($proc) { Write-Host "G0 FAIL residual q4e-server: $($proc -join ',')"; exit 1 }
Write-Host "G0 PASS: no residual q4e-server"

$body16 = '{"messages":[{"role":"user","content":"What is a cache? Explain in one sentence."}],"max_tokens":16,"temperature":0,"stream":false}'
Set-Content -Path 'body_a9_16.json' -Value $body16 -Encoding ASCII

$svr = Start-Process -FilePath 'engine\build-msvc\bin\q4e-server.exe' -ArgumentList @(
  '--model','D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
  '--host','127.0.0.1','--port','8123','--api-key','a9key',
  '--ctx-size','2048','--parallel','1'
) -RedirectStandardOutput 'a9_e2e_server.log' -RedirectStandardError 'a9_e2e_server.err.log' -NoNewWindow -PassThru
Write-Host "server pid=$($svr.Id)"

# G1: health within 420s
$deadline = (Get-Date).AddSeconds(420)
$ok = $false
while ((Get-Date) -lt $deadline) {
  $code = (curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8123/health 2>$null)
  if ($code -eq "200") { $ok = $true; break }
  if ($svr.HasExited) { break }
  Start-Sleep -Seconds 10
}
if (-not $ok) {
  Write-Host "G1 FAIL: health not 200 (exited=$($svr.HasExited)); err tail:"
  Get-Content 'a9_e2e_server.err.log' -Tail 20 -ErrorAction SilentlyContinue
  Stop-Process -Id $svr.Id -Force -ErrorAction SilentlyContinue
  exit 1
}
Write-Host "G1 PASS: health 200"

# G2: no key -> 401
$g2 = (curl.exe -s -o NUL -w "%{http_code}" -X POST http://127.0.0.1:8123/v1/chat/completions -H "Content-Type: application/json" -d "@body_c.json")
if ($g2 -eq "401") { Write-Host "G2 PASS: 401" } else { Write-Host "G2 FAIL: $g2"; Stop-Process -Id $svr.Id -Force; exit 1 }

# G3: 16-token completion
$g3 = (curl.exe -s -o a9_e2e_gen.json -w "%{http_code} time=%{time_total}" -X POST http://127.0.0.1:8123/v1/chat/completions -H "Authorization: Bearer a9key" -H "Content-Type: application/json" -d "@body_a9_16.json")
$gen = Get-Content 'a9_e2e_gen.json' -Raw -Encoding UTF8
$ct = [regex]::Match($gen, '"completion_tokens":\s*(\d+)').Groups[1].Value
if ($g3 -match '^200' -and [int]$ct -ge 1) {
  Write-Host "G3 PASS: $g3 completion_tokens=$ct"
} else {
  Write-Host "G3 FAIL: $g3 body: $gen"
  Stop-Process -Id $svr.Id -Force
  exit 1
}

# G4: SSE stream
$g4 = curl.exe -s -N -X POST http://127.0.0.1:8123/v1/chat/completions -H "Authorization: Bearer a9key" -H "Content-Type: application/json" -d "@body_d.json"
$lines = @($g4 -split "`n" | Where-Object { $_ -match '^\s*data:' })
if (($lines.Count -ge 2) -and ($g4 -match 'data:\s*\[DONE\]')) {
  Write-Host "G4 PASS: SSE $($lines.Count) data lines + [DONE]"
} else {
  Write-Host "G4 FAIL: $($lines.Count) lines; raw: $g4"
  Stop-Process -Id $svr.Id -Force
  exit 1
}

# G5: /q4e/stats + A-9 fields
$g5 = (curl.exe -s -w "`nHTTP=%{http_code}" http://127.0.0.1:8123/q4e/stats -H "Authorization: Bearer a9key") -join ' '
$ok5 = $false
if ($g5 -match 'HTTP=200' -and
    $g5 -match '"touches":\s*(\d+)' -and [int]$Matches[1] -gt 0 -and
    $g5 -match '"layers_touched":\s*(\d+)' -and [int]$Matches[1] -eq 48 -and
    $g5 -match '"hook_skips_device_ids":\s*(\d+)' -and [int]$Matches[1] -eq 0) {
  $ok5 = $true
}
if ($ok5) {
  Write-Host "G5 PASS: stats A-9 fields (touches>0, layers_touched=48, hook_skips_device_ids=0)"
} else {
  Write-Host "G5 FAIL: $g5"
  Stop-Process -Id $svr.Id -Force
  exit 1
}
Write-Host "G5 stats: $g5"

# G6: cleanup (q4e-server only - never touch the harness llama-server);
# a 111 GB process can take >3s to fully exit after SIGKILL, so wait up to 60s
Stop-Process -Name 'q4e-server' -Force -ErrorAction SilentlyContinue
$sw = [System.Diagnostics.Stopwatch]::StartNew()
while ((Get-Process -Name 'q4e-server' -ErrorAction SilentlyContinue) -and $sw.Elapsed -lt [TimeSpan]::FromSeconds(60)) {
  Start-Sleep -Milliseconds 500
}
$left = Get-Process -Name 'q4e-server' -ErrorAction SilentlyContinue
if ($left) { Write-Host "G6 FAIL residual: $($left -join ',')"; exit 1 }
Write-Host "G6 PASS: no residual q4e-server (exit after $([math]::Round($sw.Elapsed.TotalSeconds,1))s)"
Write-Host "A9_E2E_ALL_PASS"
exit 0
