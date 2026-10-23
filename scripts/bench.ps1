# Generic llama-server bench: start, warm up, tg (essay + code, thinking off), pp (~2k prompt), stop.
# usage: bench.ps1 -Name x -Model <gguf> [-Exe <llama-server.exe>] [-Extra @(...)]
# prints "[x] tg essay .. | tg code .. | pp ..", log in %TEMP%\benchm-x.log, essay text in %TEMP%\benchm-x.essay.txt
param([string]$Name, [string]$Model, [string]$Exe = "$PSScriptRoot\..\engine-v7\build-clang\bin\llama-server.exe", [string[]]$Extra, [int]$Port = 8091, [int]$N = 300)
$args0 = @('-m', $Model, '-c', '32768', '-fa', 'on', '-np', '1', '--host', '127.0.0.1', '--port', "$Port", '--no-warmup') + $Extra
$log = "$env:TEMP\benchm-$Name.log"
$p = Start-Process $Exe -ArgumentList ($args0 | % { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -WorkingDirectory (Split-Path $Exe) -RedirectStandardError $log -RedirectStandardOutput "$log.out" -PassThru -WindowStyle Hidden
$ready = $false
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep 5
    if ($p.HasExited) { break }
    try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3).status -eq 'ok') { $ready = $true; break } } catch {}
}
if (-not $ready) { "[$Name] FAILED to start"; Get-Content $log -Tail 20; if (-not $p.HasExited) { Stop-Process $p -Force }; return }
function Ask($prompt, $n) {
    $body = @{ messages = @(@{ role = 'user'; content = $prompt }); max_tokens = $n; temperature = 0; chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 5
    (Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 1800)
}
$t0 = Get-Date
$null = Ask 'Say hello.' 16
$warm = ((Get-Date) - $t0).TotalSeconds
$e = Ask 'Write a 500-word essay on the history of tea in Asia.' $N
$c = Ask 'Write a Python class implementing a thread-safe LRU cache with TTL expiry, with docstrings.' $N
$long = ('The quick brown fox jumps over the lazy dog near the river bank while the sun sets. ' * 120) + 'Summarize the text above in one sentence.'
$l = Ask $long 16
$vram = nvidia-smi --query-gpu=memory.used --format=csv,noheader
"[$Name] tg essay {0:N1} | tg code {1:N1} | pp {2:N0} t/s ({3} tok) | warm {4:N0}s | GPU VRAM {5}" -f $e.timings.predicted_per_second, $c.timings.predicted_per_second, $l.timings.prompt_per_second, $l.timings.prompt_n, $warm, $vram
$e.choices[0].message.content | Set-Content "$env:TEMP\benchm-$Name.essay.txt" -Encoding UTF8
Stop-Process $p -Force
Start-Sleep 3
