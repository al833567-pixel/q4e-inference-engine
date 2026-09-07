# Transcode ninja manifests UTF-8 -> GBK(950) so ANSI Win32 path APIs
# (FindFirstFileExA in ninja) work under ACP=950. Reversible: originals
# are kept under __ninja-utf8-backup (UTF-8).
param([string]$Mode = 'to-gbk')  # to-gbk | to-utf8
$ErrorActionPreference = 'Stop'
$root = 'build-msvc'
$bkdir = Join-Path $root '__ninja-utf8-backup'
$enc950 = [System.Text.Encoding]::GetEncoding(950)
$encUtf8 = [System.Text.Encoding]::UTF8
$files = @(
  (Join-Path $root 'build.ninja'),
  (Join-Path $root 'CMakeFiles\rules.ninja')
)
if (-not (Test-Path $bkdir)) { New-Item -ItemType Directory -Force -Path $bkdir | Out-Null }
New-Item -ItemType Directory -Force -Path (Join-Path $bkdir 'CMakeFiles') | Out-Null

foreach ($f in $files) {
  $rel = $f.Substring($root.Length + 1)
  $bk = Join-Path $bkdir $rel
  if ($Mode -eq 'to-gbk') {
    $txt = [System.IO.File]::ReadAllText($f, $encUtf8)
    $bytes = $enc950.GetBytes($txt)
    if ($enc950.GetString($bytes) -ne $txt) { throw "roundtrip failed: $f" }
    if (-not (Test-Path $bk)) { [System.IO.File]::Copy($f, $bk) }
    [System.IO.File]::WriteAllBytes($f, $bytes)
    Write-Output ("to-gbk   {0}: {1} bytes" -f $f, $bytes.Length)
  } else {
    # backup holds the original UTF-8 bytes; verify and copy back
    if (-not (Test-Path $bk)) { throw "no backup for $f" }
    $bytes = [System.IO.File]::ReadAllBytes($bk)
    $txt = $encUtf8.GetString($bytes)
    if ($encUtf8.GetBytes($txt) -ne $bytes) { throw "backup not valid UTF-8: $bk" }
    [System.IO.File]::Copy($bk, $f, $true)
    Write-Output ("to-utf8  {0}: {1} bytes (restored from backup)" -f $f, $bytes.Length)
  }
}
Write-Output 'done'
