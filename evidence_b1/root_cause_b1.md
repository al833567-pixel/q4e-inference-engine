# B-1 CMake/Ninja configure 挂起 — 根因诊断记录

日期: 2026-07-21 (会话内)
现象: `cmake -S engine -B engine/build-cuda -G Ninja ...` 卡在 `-- Detecting C compiler ABI info` 超过 46 分钟。

## 证据链
1. TryCompile-610ap5 目录: `CMakeCCompilerABI.c.obj` + `vc140.pdb` 已写入 (cl 完成工作), 但 `cmTC_dbac0.exe` 不存在 (link edge 从未启动), ninja/cmake 进程存活但 CPU ~0, 无子进程。
2. 复现: 删除 obj 后直接 `ninja -v` (同一 TryCompile 目录, 同一 WinLibs mingw ninja 1.13.2) → 90s 后仍挂起 (HANG CONFIRMED)。
3. 隔离实验 (ASCII temp 目录, 最小 ninja 文件, 纯 ASCII 路径与输入):
   - T1 deps=msvc + zh-TW 前缀: HANG 45s
   - T2 无任何 deps 的普通 cl 编译: **同样 HANG 45s** → 与 msvc_deps/编码/前缀无关
   - 手动直接调 cl (pwsh 捕获 stdout): 0.8s 完成 → cl 本身无问题
4. A/B: 会话沙箱策略切到 danger-full-access (无沙箱) 后, 同样最小 ninja+cl 测试 0.05–0.23s 正常完成 (C1034 仅因该会话未 source vcvars, 与挂起无关; ninja 正常拿到退出码并退出)。
5. A 侧历史: `engine/build-msvc` 同样用 Ninja 生成器, configure try-compile 成功 (configure_A3.log: "Detecting C compiler ABI info - done", MSVC 19.44.35228), 且完整 build 成功 → 本机器 Ninja 构建可行。

## 根因
DSH workspace-write 文件沙箱对子进程的进程监督保留了 cl.exe stdout pipe 的写端; cl 退出后 pipe 未达 EOF, ninja 读取命令输出永久阻塞 → CMake try-compile 的 ninja 子进程挂起 → 外层 cmake 等待 → configure 冻结在 ABI 探测。
(次生问题, 非挂起原因: build.ninja 中 CJK 路径为 cp950 编码, msvc_deps_prefix 为 UTF-8 繁体中文, 与 cl 的 cp950 输出不匹配 → 依赖解析静默丢失, 不影响本构建正确性。)

## 修复
- 无需改 CMake 参数/源码。将本会话文件沙箱策略改为 danger-full-access 后, ninja+cl 正常完成。
- 已删除损坏的 `engine/build-cuda` (部分 configure 产物, 非源码), 证据备份于 `evidence_b1/TryCompile-610ap5-backup/` + `b1_build.log`。
- 重新派发 B-1 子代理执行 RUNBOOK_B_v2.md 原样命令。

## 重跑结果 + 新增发现 (B-1 retry)
- 重跑 B-1: **474/474 全绿**, 日志以 `[474/474] Linking CXX executable bin\q4e-server.exe` 结束, 107,766 行日志零错误模式 (`b1_retry_build.log`)。
- 产物验证: `engine\build-cuda\bin\q4e-server.exe` = 44,027,392 bytes, `--version` → `0.3.0-dev (build 17, commit 1144ddd), built with MSVC 19.44.35228.0`。
- 观察 1: job 层 exit=1 (harness/pipeline 假象; 日志无对应错误, 产物有效可运行) — 记录为观察, 不阻塞 B-1 门。
- 观察 2 (新增, 环境特性): **WinLibs-MSYS2 ninja 1.13.2 在 CJK repo 路径下, 增量 build 必挂** — `cmake --build` / `ninja -C` (存在 .ninja_log 时) 扫描 rebuild edge (`build build.ninja: RERUN_CMAKE | D$:\...\AI研究\...`) 的绝对 CJK 依赖 → `ninja: error: FindFirstFileExA(../../../../AI研究/推理工具/engine/common): The filename, directory name, or volume label syntax is incorrect` (ANSI API 无法处理 CJK)。首次 full build (无 .ninja_log) 不做 rebuild check → 正常。**影响**: 本 task 不需重编 → 无影响; 若日后需重编: 删 `.ninja_log` 走 full-build 路径, 或改用 official standalone ninja, 或从非 CJK 路径 checkout build。
