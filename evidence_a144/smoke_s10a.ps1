# S10a (A-14.4) live smoke: ngram (PLE) prefetch on the q4e live server
# Spawns q4e-server itself, runs the 5 probes, kills ONLY the spawned PID.
$ErrorActionPreference = "Stop"

# paths come from env (the child powershell may not decode a UTF-8 script's non-ASCII literals)
if (-not $env:S10A_ROOT -or -not $env:S10A_MODEL) { Write-Output "SMOKE_FAIL env: S10A_ROOT / S10A_MODEL not set"; exit 1 }
$root   = $env:S10A_ROOT
$ev     = Join-Path $root "evidence_a144"
$bin    = Join-Path $root "engine\build-msvc\bin\q4e-server.exe"
$model  = $env:S10A_MODEL
$host_  = "127.0.0.1"
$port   = 8131
$key    = "bkey"
$base   = "http://${host_}:${port}"

New-Item -ItemType Directory -Force -Path $ev | Out-Null
$serverLog = Join-Path $ev "server.log"

# G0: no residual q4e-server, port must be free
$pre = Get-Process -Name "q4e-server" -ErrorAction SilentlyContinue
if ($pre) { Write-Output "SMOKE_FAIL G0: residual q4e-server process(es) found"; exit 1 }
try {
    $r = Invoke-WebRequest -Uri "${base}/health" -TimeoutSec 2 -UseBasicParsing
    Write-Output "SMOKE_FAIL G0: port $port already serving ($($r.StatusCode))"; exit 1
} catch { } # port free (or nothing listening) - ok

# START: spawn the server (only PID we will ever kill)
$proc = Start-Process -FilePath $bin -ArgumentList @(
    "--model", $model,
    "--host", $host_,
    "--port", "$port",
    "--api-key", $key,
    "--ctx-size", "2048",
    "--parallel", "1"
) -RedirectStandardOutput (Join-Path $ev "server.out.log") `
 -RedirectStandardError  (Join-Path $ev "server.err.log") -PassThru
Write-Output "START pid=$($proc.Id)"

function Stop-Server {
    param($p)
    if ($p -and -not $p.HasExited) {
        $p.Kill() | Out-Null
        $p.WaitForExit(180000) | Out-Null
    }
    $still = Get-Process -Id $p.Id -ErrorAction SilentlyContinue
    if ($still) { Write-Output "KILL: pid $($p.Id) still alive after 180 s"; exit 1 }
    Write-Output "KILL: pid $($p.Id) exited"
}

function Get-Stats {
    $r = Invoke-RestMethod -Uri "${base}/q4e/stats" -Headers @{ Authorization = "Bearer ${key}" }
    return $r
}

function Post-Completion {
    param($prompt, $outFile)
    $body = @{ model = "q4e"; prompt = $prompt; max_tokens = 16; temperature = 0 } | ConvertTo-Json
    $r = Invoke-WebRequest -Uri "${base}/v1/completions" -Method POST `
        -Headers @{ Authorization = "Bearer ${key}"; "Content-Type" = "application/json" } `
        -Body $body -UseBasicParsing
    $r.Content | Set-Content -Path $outFile -Encoding UTF8
    return $r
}

function Set-NgramMode {
    param($mode, $outFile)
    $body = @{ mode = $mode } | ConvertTo-Json
    $r = Invoke-WebRequest -Uri "${base}/q4e/ngram" -Method POST `
        -Headers @{ Authorization = "Bearer ${key}"; "Content-Type" = "application/json" } `
        -Body $body -UseBasicParsing
    $r.Content | Set-Content -Path $outFile -Encoding UTF8
    return $r
}

# G1: wait for /health (deadline 480 s)
$deadline = (Get-Date).AddSeconds(480)
$healthy = $false
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) {
        Write-Output "SMOKE_FAIL G1: server exited early (code $($proc.ExitCode)); tail:"
        Get-Content (Join-Path $ev "server.err.log") -Tail 20 -ErrorAction SilentlyContinue
        Stop-Server $proc
        exit 1
    }
    try {
        $h = Invoke-WebRequest -Uri "${base}/health" -TimeoutSec 5 -UseBasicParsing
        if ($h.StatusCode -eq 200) { $healthy = $true; break }
    } catch { }
    Start-Sleep -Seconds 5
}
if (-not $healthy) {
    Write-Output "SMOKE_FAIL G1: /health not ready within 480 s; tail:"
    Get-Content (Join-Path $ev "server.err.log") -Tail 20 -ErrorAction SilentlyContinue
    Stop-Server $proc
    exit 1
}
Write-Output "G1: /health ok"

try {
    # P1: initial stats
    $s0 = Get-Stats
    $s0 | Set-Content -Path (Join-Path $ev "stats_p1.json") -Encoding UTF8
    Write-Output ("P1: ngram_prefetch={0} ngram_hints={1} ngram_rows_warmed={2}" -f $s0.ngram_prefetch, $s0.ngram_hints, $s0.ngram_rows_warmed)
    if ($s0.ngram_prefetch -ne 2) { Write-Output "SMOKE_FAIL P1: ngram_prefetch != 2"; exit 1 }
    if ($s0.ngram_hints -ne 0) { Write-Output "SMOKE_FAIL P1: ngram_hints != 0"; exit 1 }

    # P2: 16 tokens in full mode -> hints >= 15, rows warmed > 0
    Post-Completion "List primes from 2, one per line." (Join-Path $ev "gen_p2.json") | Out-Null
    Start-Sleep -Seconds 3
    $s1 = Get-Stats
    $s1 | Set-Content -Path (Join-Path $ev "stats_p2.json") -Encoding UTF8
    Write-Output ("P2: ngram_hints={0} ngram_rows_warmed={1} ngram_rows_dropped={2} ngram_bytes_warmed={3}" -f $s1.ngram_hints, $s1.ngram_rows_warmed, $s1.ngram_rows_dropped, $s1.ngram_bytes_warmed)
    if ($s1.ngram_hints -lt 15) { Write-Output "SMOKE_FAIL P2: ngram_hints < 15"; exit 1 }
    if ($s1.ngram_rows_warmed -le 0) { Write-Output "SMOKE_FAIL P2: ngram_rows_warmed <= 0"; exit 1 }

    # P3: shim mode -> ok, hints grow, rows_warmed unchanged
    $m3 = Set-NgramMode "shim" (Join-Path $ev "mode_p3.json")
    Write-Output ("P3: mode resp: {0}" -f $m3.Content)
    if ($m3.Content -notmatch '"ok":\s*true') { Write-Output "SMOKE_FAIL P3: shim mode not ok"; exit 1 }
    Post-Completion "Explain the capital of France in one sentence." (Join-Path $ev "gen_p3.json") | Out-Null
    Start-Sleep -Seconds 3
    $s2 = Get-Stats
    $s2 | Set-Content -Path (Join-Path $ev "stats_p3.json") -Encoding UTF8
    Write-Output ("P3: ngram_hints={0} ngram_rows_warmed={1}" -f $s2.ngram_hints, $s2.ngram_rows_warmed)
    if ($s2.ngram_hints -le $s1.ngram_hints) { Write-Output "SMOKE_FAIL P3: hints did not grow in shim"; exit 1 }
    if ($s2.ngram_rows_warmed -ne $s1.ngram_rows_warmed) { Write-Output "SMOKE_FAIL P3: rows_warmed changed in shim"; exit 1 }

    # P4: back to full with a NEW prompt -> rows_warmed grows
    $m4 = Set-NgramMode "on" (Join-Path $ev "mode_p4.json")
    Write-Output ("P4: mode resp: {0}" -f $m4.Content)
    if ($m4.Content -notmatch '"ok":\s*true') { Write-Output "SMOKE_FAIL P4: on mode not ok"; exit 1 }
    Post-Completion "Name the seven colors of the rainbow." (Join-Path $ev "gen_p4.json") | Out-Null
    Start-Sleep -Seconds 3
    $s3 = Get-Stats
    $s3 | Set-Content -Path (Join-Path $ev "stats_p4.json") -Encoding UTF8
    Write-Output ("P4: ngram_hints={0} ngram_rows_warmed={1} ngram_bytes_warmed={2}" -f $s3.ngram_hints, $s3.ngram_rows_warmed, $s3.ngram_bytes_warmed)
    if ($s3.ngram_hints -le $s2.ngram_hints) { Write-Output "SMOKE_FAIL P4: hints did not grow"; exit 1 }
    if ($s3.ngram_rows_warmed -le $s2.ngram_rows_warmed) { Write-Output "SMOKE_FAIL P4: rows_warmed did not grow"; exit 1 }

    # P5: log line
    $logtext = (Get-Content (Join-Path $ev "server.out.log") -Raw -ErrorAction SilentlyContinue) + "`n" + (Get-Content (Join-Path $ev "server.err.log") -Raw -ErrorAction SilentlyContinue)
    if ($logtext -notmatch "ngram prefetch ready") {
        Write-Output "SMOKE_FAIL P5: log missing 'ngram prefetch ready'"; exit 1
    }
    $line = ($logtext -split "`n" | Select-String "ngram prefetch ready" | Select-Object -First 1).ToString().Trim()
    Write-Output ("P5: log line: {0}" -f $line)

    Write-Output "SMOKE_PASS"
} finally {
    Stop-Server $proc
}
