# q4e-e2e

End-to-end acceptance client for `q4e-server` (C++ replacement of the legacy
`a9_e2e.ps1`). Spawns a `q4e-server` child with a real GGUF, runs gates
G0-G6, then kills only the child PID it started (never any other process).

Requires a `q4e-server` build and a GGUF model on disk. Links only vendored
deps (`cpp-httplib`, `nlohmann`).

## Usage

```
q4e-e2e.exe --server engine\build-msvc\bin\q4e-server.exe \
            --model <gguf-path> --log-dir evidence_a141
```

## Arguments

| arg                | default      | meaning                                   |
|--------------------|--------------|-------------------------------------------|
| `--server <path>`  | (required)   | path to q4e-server binary                 |
| `--model <path>`   | (required)   | path to GGUF model file                   |
| `--host <ip>`      | `127.0.0.1`  | bind/connect host                         |
| `--port <n>`       | `8123`       | server port                               |
| `--api-key <key>`  | `a141key`    | bearer key passed to server and requests  |
| `--ctx-size <n>`   | `2048`       | server `--ctx-size`                       |
| `--max-tokens <n>` | `64`         | G3 completion `max_tokens`                |
| `--health-timeout <s>` | `480`    | G1 deadline in seconds (poll every 5 s)   |
| `--log-dir <dir>`  | `q4e-e2e-run`| evidence dir, created if missing          |

The log dir receives: `server.out.log`, `server.err.log`, `gen.json`,
`sse.txt`, `stats.json`.

## Gates

- **G0** no residual server process (Windows: process-name scan;
  non-Windows: skipped) and the port is free (HTTP GET with 2 s timeout).
- **START** spawn `q4e-server` with
  `--model ... --host ... --port ... --api-key ... --ctx-size ... --parallel 1`,
  stdout/stderr redirected into the log dir.
- **G1** `GET /health` returns 200 within `--health-timeout`; fail (with
  `server.err.log` tail) if the child exits first.
- **G2** `POST /v1/chat/completions` without auth -> 401.
- **G3** authed completion (64 `max_tokens`, `temperature` 0) -> 200 and
  `usage.completion_tokens >= 1`; body saved to `gen.json`.
- **G4** authed stream request (8 tokens) -> `>= 2` `data:` lines and one
  `data: [DONE]`; body saved to `sse.txt`.
- **G5** authed `GET /q4e/stats` -> `q4e_prefetch == 1`, `touches > 0`,
  `layers_touched == expected_touched` (and `layers > 0`),
  `prefetch_io_issued > 0`, `prefetch_overlap_hits > 0`; other prefetch/hook
  fields are printed, not gated. Raw body saved to `stats.json`.
  `expected_touched` is placement-aware: the hook books host-resident MoE
  layers only, so `expected = layers` when no layer is device-resident
  (`hook_skips_weights_device == 0`, e.g. CPU run) and
  `expected = layers - q4e_moe_vram` (or `0` when `q4e_moe_vram >= layers`)
  when device-resident layers exist.
- **G6** `TerminateProcess`/`SIGKILL` only the spawned PID, wait up to 180 s
  for exit (a ~100 GB process can linger after kill), then re-scan
  (Windows) to confirm the PID is gone.

## Result

- success: prints `Q4E_E2E_ALL_PASS`, exit code 0
- failure: prints `<gate> FAIL: <reason>` + `Q4E_E2E_FAIL <gate>`, exit code 1
  (the child, if already started, is killed before exit)
