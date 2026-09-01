# q4e Provenance

- **Upstream**: https://github.com/ggml-org/llama.cpp
- **Commit**: `458681e1d5d4a29a1463c4732e03226cf384b997`（tag `b10729`，2026-08-31）
- **Clone**: 2026-09-01（Task A，A-1）；`--depth 1`
- **License**: MIT（見 `LICENSE`）
- **qwen4exp**: 上游**原生收**（`src/models/qwen4exp.cpp`、`LLM_ARCH_QWEN4EXP`、PLE/QSA 完整實作）→ A-2 只需 CPU 冒煙驗證
- **改名**: project `llama.cpp` → `q4e`；可執行檔 `llama-server` → `q4e-server`
- **擁有權**: 本樹＝q4e 自有 source；`D:\llama` fork＝oracle only（其代碼不進本樹）
- **內嵌 .git 已移除**: 本快照以 `..` repo 的 git 為歷史；需對照上游時用上面 hash 重 clone
