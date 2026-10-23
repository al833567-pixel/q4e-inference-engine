# A/B runs of Flash-Next with the same start: the reference expert profile (<model>.q4e-profile.ref, a copy of a
# profile you want to start from) is restored before each run.
# usage: ab.ps1 -ModelDir <folder> -Runs @(@{name='a'; env=@{X='1'}; extra=@()}, ...) [-Mtp 0|2|3] [-Repeat 2]
param([string]$ModelDir, [object[]]$Runs, [int]$Mtp = 0, [int]$Repeat = 1,
      [string]$Exe = "$PSScriptRoot\..\engine-v7\build-clang\bin\llama-server.exe")
$bench = "$PSScriptRoot\bench.ps1"
$m     = "$ModelDir\Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"
$base  = @('--device', 'CUDA0', '-ngl', '99', '--q4e', '-lv', '4', '-t', '8')
if ($Mtp -gt 0) { $base += @('--spec-type', 'draft-mtp', '-md', "$ModelDir\mtp-Qwen3.8-Flash-Next-Q4_0.gguf", '-ngld', '99', '--spec-draft-n-max', "$Mtp") }
for ($r = 1; $r -le $Repeat; $r++) {
    foreach ($run in $Runs) {
        if (Test-Path "$m.q4e-profile.ref") {
            Copy-Item "$m.q4e-profile.ref" "$m.q4e-profile" -Force
        }
        $saved = @{}
        foreach ($k in $run.env.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $run.env[$k]) }
        & $bench -Name "$($run.name)-$r" -Exe $Exe -Model $m -Extra ($base + $run.extra) 2>&1 | Select-String '^\[' | % Line
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    }
}
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
