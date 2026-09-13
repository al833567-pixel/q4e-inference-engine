# q4e hit-rate experiment (CPU): single-type vs different-type, capacity-aware
# Tiers: 24GB VRAM = 13 layers (100% hit); remaining 35 layers = RAM/SSD (hit rate measured here)
# Feasibility (single user + similar types): speedup (ON>OFF) AND no visible SSD impact.
$ErrorActionPreference = 'Continue'
$server  = 'D:\q4e_build\build\bin\q4e-server.exe'
$model   = 'D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
$base    = 'D:\q4e_hitrate'
$port    = 8124
$apiKey  = 'bench-key'
$host_   = "http://127.0.0.1:$port"

New-Item -ItemType Directory -Force -Path $base | Out-Null
function Log($m) { $line = (Get-Date -Format 'HH:mm:ss') + " " + $m; Write-Host $line; Add-Content -Path "$base\run.log" -Value $line }

function Port-Free([int]$p) {
    $dl = (Get-Date).AddSeconds(120)
    while ((Get-Date) -lt $dl) {
        try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1', $p); $c.Close(); Start-Sleep -Seconds 2 }
        catch { return $true }
    }
    return $false
}
function Start-Server([bool]$prefetchOn, [string]$tag) {
    if (-not (Port-Free $port)) { Log "[$tag] port $port still in use, abort"; return $null }
    $a = @('-m', $model, '--host', '127.0.0.1', '--port', "$port", '--api-key', $apiKey,
           '--n-gpu-layers', '0', '--q4e-moe-vram', '10')
    if ($prefetchOn) { $a += @('--q4e-warm-min-freq', '2') } else { $a += @('--no-q4e-prefetch') }
    Log "[$tag] START args: $($a -join ' ')"
    $p = Start-Process -FilePath $server -ArgumentList $a -PassThru `
         -RedirectStandardOutput "$base\$tag.out" -RedirectStandardError "$base\$tag.err"
    return $p
}
function Stop-Server($proc, [string]$tag) {
    if ($proc -and -not $proc.HasExited) {
        Log "[$tag] KILL pid=$($proc.Id)"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        $proc.WaitForExit(15000) | Out-Null
        if (-not $proc.HasExited) { Log "[$tag] KILL failed, still alive" } else { Log "[$tag] KILL ok" }
    }
}
function Wait-Health([string]$tag) {
    $dl = (Get-Date).AddSeconds(900)
    while ((Get-Date) -lt $dl) {
        try { Invoke-RestMethod -Uri "$host_/health" -Method Get -TimeoutSec 5 | Out-Null; Log "[$tag] HEALTH ok"; return $true }
        catch { Start-Sleep -Seconds 5 }
    }
    Log "[$tag] HEALTH TIMEOUT"; return $false
}
function Send-Chat([string]$prompt, [int]$maxTok, [string]$tag) {
    $body = @{ messages = @(@{ role = 'user'; content = $prompt }); max_tokens = $maxTok; temperature = 0; stream = $false } | ConvertTo-Json -Depth 6
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $r = Invoke-RestMethod -Uri "$host_/v1/chat/completions" -Method Post -Body $body `
              -ContentType 'application/json' -Headers @{ Authorization = "Bearer $apiKey" } -TimeoutSec 2400
        $sw.Stop()
        $tok = 0; if ($r.usage -and $r.usage.completion_tokens) { $tok = [int]$r.usage.completion_tokens }
        $tps = if ($sw.Elapsed.TotalSeconds -gt 0) { [math]::Round($tok / $sw.Elapsed.TotalSeconds, 4) } else { 0 }
        Log "[$tag] CHAT ok tokens=$tok elapsed=$([math]::Round($sw.Elapsed.TotalSeconds,1))s tps=$tps"
        return @{ ok = $true; tokens = $tok; elapsed = $sw.Elapsed.TotalSeconds; tps = $tps }
    } catch {
        $sw.Stop()
        Log "[$tag] CHAT FAIL: $($_.Exception.Message)"
        return @{ ok = $false; tokens = 0; elapsed = $sw.Elapsed.TotalSeconds; tps = 0; err = $_.Exception.Message }
    }
}
function Get-Stats([string]$tag) {
    try {
        $s = Invoke-RestMethod -Uri "$host_/q4e/stats" -Method Get -Headers @{ Authorization = "Bearer $apiKey" } -TimeoutSec 30
        $s | ConvertTo-Json -Depth 6 | Out-File "$base\$tag_stats.json" -Encoding utf8
        $tot = [long]$s.touches
        $fast = [long]$s.hits_vram + [long]$s.hits_ram
        $ssd  = [long]$s.hits_ssd
        $fastPct = if ($tot -gt 0) { [math]::Round(100.0 * $fast / $tot, 2) } else { 0 }
        $ssdPct  = if ($tot -gt 0) { [math]::Round(100.0 * $ssd / $tot, 2) } else { 0 }
        Log "[$tag] STATS touches=$tot hits_vram=$($s.hits_vram) hits_ram=$($s.hits_ram) hits_ssd=$ssd fastPct=$fastPct ssdPct=$ssdPct overlap=$($s.prefetch_overlap_ratio)"
        return $s
    } catch {
        Log "[$tag] STATS FAIL: $($_.Exception.Message)"; return $null
    }
}

$singlePrompt = 'Write an essay about caching in computer systems, covering L1/L2 caches, CPU cache hierarchy, cache coherence, and write policies.'
$diffPrompts = @(
  'Explain how the quicksort algorithm works, then write a clean C++ implementation with partial pivoting.',
  'Translate this French sentence into natural English and explain the grammar: "Malgré la pluie, nous avons continué notre randonnée."',
  'Prove that the square root of 2 is irrational using a proof by contradiction, step by step.',
  'Explain the theory of general relativity in simple terms, including the curvature of spacetime and gravitational time dilation.',
  'Write a short haiku and a short sonnet, both about autumn leaves falling.'
)

# ============ Experiment A: ON, single-type (fixed-type) ============
Log "===== EXPERIMENT A: ON single-type (fixed-type) ====="
$pa = Start-Server $true 'A_on_single'
if ($pa -and (Wait-Health 'A')) {
  $ra = Send-Chat $singlePrompt 256 'A'
  Get-Stats 'A_on_single' | Out-Null
}
Stop-Server $pa 'A'
Start-Sleep -Seconds 10

# ============ Experiment B: ON, different-type ============
Log "===== EXPERIMENT B: ON different-type (5 types) ====="
$pb = Start-Server $true 'B_on_diff'
if ($pb -and (Wait-Health 'B')) {
  foreach ($i in 0..4) {
    $rb = Send-Chat $diffPrompts[$i] 48 "B[$i]"
    if (-not $rb.ok) { break }
  }
  Get-Stats 'B_on_diff' | Out-Null
}
Stop-Server $pb 'B'
Start-Sleep -Seconds 10

# ============ Experiment C: OFF, single-type (baseline t/s) ============
Log "===== EXPERIMENT C: OFF single-type (baseline) ====="
$pc = Start-Server $false 'C_off_single'
if ($pc -and (Wait-Health 'C')) {
  $rc = Send-Chat $singlePrompt 256 'C'
}
Stop-Server $pc 'C'

Log "===== ALL DONE ====="
