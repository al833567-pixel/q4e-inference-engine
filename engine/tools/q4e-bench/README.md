# q4e-bench

ON/OFF benchmark client for `q4e-server` (C++ replacement of the legacy
`bench_tps.py --measure/--prefetch` plus the B-8-4/B-8-6 manual curl steps).
Runs the ON phase (prefetch on) then the OFF phase (oracle, or `q4e-server`
with `--no-q4e-prefetch` and the same placement), strictly serial: the ON
server is killed and confirmed gone before OFF starts. Each phase spawns its
own server child and kills only that child PID (never any other process,
including any running `llama-server`).

Measurement discipline (RUNBOOK_B B-8-4): fixed prompt, `temp=0`, one warmup
completion (`--warmup-tokens`) + one measured completion (`--measure-tokens`),
`tps = usage.completion_tokens / time_total`. This fork does not send a
top-level `time_total` (it sends `timings`), so `time_total` is the client-side
wall time of the whole request - the same quantity the RUNBOOK measured with
`curl -w "%{time_total}"` (includes prefill + decode).

Links only vendored deps (`cpp-httplib`, `nlohmann`).

## Usage

```
q4e-bench.exe --server engine\build-msvc\bin\q4e-server.exe ^
              --model <gguf-path> --warmup-tokens 16 --measure-tokens 32 ^
              --log-dir evidence_a143
```

## Arguments

| arg                    | default | meaning |
|------------------------|---------|---------|
| `--server <path>`      | (required) | path to q4e-server binary |
| `--model <path>`       | (required) | path to GGUF model file |
| `--host <ip>`          | `127.0.0.1` | bind/connect host |
| `--port-on <n>`        | `8124`  | ON phase port |
| `--port-off <n>`       | `8125`  | OFF phase port |
| `--api-key <key>`      | `bkey`  | bearer key for server + requests |
| `--ctx-size <n>`       | `2048`  | server `--ctx-size` |
| `--warmup-tokens <n>`  | `256`   | warmup completion `max_tokens` |
| `--measure-tokens <n>` | `256`   | measured completion `max_tokens` |
| `--prompt <text>`      | essay prompt | fixed prompt for both runs of a phase |
| `--gpu-layers <v>`     | `all`   | server `--n-gpu-layers` value |
| `--moe-vram <n>`       | `10`    | server `--q4e-moe-vram` value (placement K) |
| `--oracle <path>`      | (unset) | `llama-server` for the OFF phase; unset = q4e + `--no-q4e-prefetch` |
| `--oracle-ncmoe <n>`   | `38`    | oracle `-ncmoe` value |
| `--health-timeout <s>` | `480`   | health poll deadline (poll every 5 s) |
| `--log-dir <dir>`      | `q4e-bench-run` | evidence dir, created if missing |
| `--phases <set>`       | `both`  | `both`, `on`, or `off` |

The log dir receives: `on_server.out.log`, `on_server.err.log`,
`on_warmup.json`, `on_measure.json`, `on_stats.json`, the matching `off_*`
files (`off_stats.json` only when OFF is the q4e server), and
`bench_report.json`.

## Phases

- **ON** q4e-server with `--n-gpu-layers <gpu-layers> --q4e-moe-vram <moe-vram>`
  (prefetch on) on `--port-on`.
- **OFF** if `--oracle` is set: that `llama-server` with
  `-m <model> --host <host> --port <port-off> -ngl all -ncmoe <oracle-ncmoe>
  --ctx-size <ctx>` (no `/q4e/stats`); otherwise the same q4e-server with the
  same placement args plus `--no-q4e-prefetch` on `--port-off`.

Serial redline: never two model servers at once; OFF starts only after ON is
killed and confirmed gone.

## Gates (B-8-6, GPU-side)

- warm t/s(ON) >= t/s(OFF)
- warm t/s(ON) >= 1.95
- `prefetch_overlap_ratio` >= 0.80 (from the ON `/q4e/stats`)

`verdict` is `PASS` iff both phases ran and all three gates hold; `INCOMPLETE`
if a phase or the ON stats is missing; otherwise `FAIL`. A perf `FAIL` is a
normal result on a CPU box - it is recorded in `bench_report.json` and the
tool still exits 0.

## Result

- mechanical success: prints a summary table + `Q4E_BENCH_DONE verdict=<V>`,
  exit 0
- mechanical failure: prints `Q4E_BENCH_FAIL <phase>: <reason>`, exit 1
  (any already-started child is killed before exit)
