# q4e-ngram

Ngram (PLE table) prefetch diag/accept client for `q4e-server`. C++ replacement
of the legacy `ngram_diag.py` (4-stage attribution A/B/C/D) and
`ngram_accept.py` (P1-P4 drop gate, RUNBOOK_B section 5). One server instance
per run; phases are strictly serial and the ngram mode is switched per phase
via `POST /q4e/ngram` (`{"mode":"off"|"shim"|"full"}`). The tool only ever
kills the child PID it spawned itself - never any other process (including a
running `llama-server`). Never two model instances at once.

Measurement discipline: fixed prompt, `temp=0`, one non-stream warmup
completion, then per phase one SSE-streamed `POST /v1/completions` with
per-chunk client timestamps (httplib cannot stream responses, so timed phases
use raw sockets). One arrival is counted per `data:` chunk, matching the
legacy per-chunk stamping. `cum = N / wall`; `tail = TAIL / span of the last
TAIL+1 timestamps` (falls back to cum when fewer).

Links only vendored deps (`cpp-httplib`, `nlohmann`) + `winsock2`.

## Usage

```
q4e-ngram.exe diag   --server engine\build-msvc\bin\q4e-server.exe ^
                     --model <gguf-path> --warmup-tokens 16 --tokens 32 ^
                     --log-dir evidence_a144\diag

q4e-ngram.exe accept --server engine\build-msvc\bin\q4e-server.exe ^
                     --model <gguf-path> --warmup-tokens 16 --tokens 48 ^
                     --tail 16 --log-dir evidence_a144\accept
```

## Arguments

| arg                  | default | meaning |
|----------------------|---------|---------|
| `<diag\|accept>`     | (required, positional) | mode: 4-stage attribution, or P1-P4 drop gate |
| `--server <path>`    | (required) | path to q4e-server binary |
| `--model <path>`     | (required) | path to GGUF model file |
| `--host <ip>`        | `127.0.0.1` | bind/connect host |
| `--port <n>`         | `8132`  | server port |
| `--api-key <key>`    | `bkey`  | bearer key for server + requests |
| `--ctx-size <n>`     | `2048`  | server `--ctx-size` |
| `--warmup-tokens <n>`| `300`   | warmup completion `max_tokens` |
| `--tokens <n>`       | `300`   | per-phase `max_tokens` |
| `--tail <n>`         | `150`   | accept tail window (last N tokens) |
| `--prompt <text>`    | prime prompt | fixed prompt: `List prime numbers starting from 2, one number per line, no explanations, keep going.` |
| `--gpu-layers <v>`   | `all`   | server `--n-gpu-layers` value (warn-and-ignore on CPU) |
| `--moe-vram <n>`     | `10`    | server `--q4e-moe-vram` value (warn-and-ignore on CPU) |
| `--health-timeout <s>`| `480`  | health poll deadline (poll every 5 s) |
| `--log-dir <dir>`    | `q4e-ngram-run` | evidence dir, created if missing |

The log dir receives: `server.out.log`, `server.err.log`, `warmup.json`,
per-phase `<name>_mode.json` (mode-set response), `<name>_sse.txt` (raw SSE
body), per-full-phase-in-accept `<name>_before_stats.json` /
`<name>_after_stats.json`, `final_stats.json`, and `<mode>_report.json`.

## Modes

- **diag** `A off -> B shim -> C full -> D full`: isolates executor/shim cost
  (B) vs full row-read cost (C) vs stability (D); prints the rolling-50
  series like the legacy tool. Verdict `n/a` (attribution, no gate).
- **accept** `P1 off -> P2 full -> P3 off -> P4 full`: position-balanced
  pairs to cancel page-cache drift. `drop_cum = (off_cum - full_cum)/off_cum`,
  `drop_tail` likewise on the tail window; also reports the
  `ngram_rows_warmed` delta per full phase (rows/token prefetch volume).
  **PASS iff max of the 4 drops < 1.0%** (negative drop = speedup = fine).
  The perf verdict is a B-side GPU gate: on a CPU box it is data, not a
  failure.

## Redlines

- phases strictly serial on ONE server instance; next phase starts only after
  the previous timed stream completed
- only the child PID this tool spawned is ever killed (wait up to 180 s,
  confirm gone by process snapshot)
- a mode-set response that is not `{"ok":true,...}` aborts the run as a
  mechanical failure

## Result

- mechanical success: prints a summary table + `Q4E_NGRAM_DONE verdict=<V>`,
  exit 0 (a `FAIL` verdict is perf data in `<mode>_report.json`)
- mechanical failure: prints `Q4E_NGRAM_FAIL <phase>: <reason>`, exit 1
  (any already-started child is killed before exit)
