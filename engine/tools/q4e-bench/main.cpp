// q4e-bench: ON/OFF benchmark client for q4e-server
// replaces legacy_v1/tools/bench_tps.py --measure/--prefetch + the B-8-4/B-8-6 curl steps

#include "q4e-bench-v6.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <tlhelp32.h>
    #include <string.h>
#else
    #include <fcntl.h>
    #include <iterator>
    #include <netdb.h>
    #include <signal.h>
    #include <sys/socket.h>
    #include <sys/time.h>
    #include <sys/wait.h>
    #include <unistd.h>
#endif

struct conf {
    std::string server;
    std::string model;
    std::string host = "127.0.0.1";
    int         port_on = 8124;
    int         port_off = 8125;
    std::string api_key = "bkey";
    int         ctx_size = 2048;
    int         warmup_tokens = 256;
    int         measure_tokens = 256;
    std::string prompt = "Write an essay about caching in computer systems.";
    std::string gpu_layers = "all";
    int         moe_vram = 10;
    int         warm_min_freq = 1; // ON phase server --q4e-warm-min-freq (1 = legacy, warm all)
    std::string oracle;
    int         oracle_ncmoe = 38;
    int         health_timeout = 480;
    std::string log_dir = "q4e-bench-run";
    std::string phases = "both";
    bool        v6 = false;      // run the v6 component micro-benchmark and exit
    bool        v6_kv100k = false; // v6 kv section: 300-step sliding window over 100k ctx
};

struct child {
    bool started = false;
#ifdef _WIN32
    HANDLE process = nullptr;
    DWORD  pid = 0;
#else
    pid_t  pid = 0;
#endif
};

struct phase_result {
    bool         ran = false;
    bool         prefetch = false;
    std::string  server;
    long long    completion_tokens = 0;
    double       time_total = 0.0;
    double       tps = 0.0;
    bool         has_stats = false;
    nlohmann::json stats;
};

struct http_result {
    bool        ok = false;
    int         status = 0;
    std::string body;
};

static void usage() {
    printf("q4e-bench: ON/OFF benchmark client for q4e-server\n");
    printf("(replaces bench_tps.py --measure/--prefetch + the B-8-4/B-8-6 curl steps)\n");
    printf("\n");
    printf("usage: q4e-bench --server <q4e-server-exe> --model <gguf> [options]\n");
    printf("\n");
    printf("required:\n");
    printf("  --server <path>      path to q4e-server binary\n");
    printf("  --model <path>       path to GGUF model file\n");
    printf("\n");
    printf("options:\n");
    printf("  --host <ip>          default: 127.0.0.1\n");
    printf("  --port-on <n>        ON phase port, default: 8124\n");
    printf("  --port-off <n>       OFF phase port, default: 8125\n");
    printf("  --api-key <key>      bearer key for server + requests, default: bkey\n");
    printf("  --ctx-size <n>       server --ctx-size, default: 2048\n");
    printf("  --warmup-tokens <n>  warmup completion max_tokens, default: 256\n");
    printf("  --measure-tokens <n> measured completion max_tokens, default: 256\n");
    printf("  --prompt <text>      default: Write an essay about caching in computer systems.\n");
    printf("  --gpu-layers <v>     server --n-gpu-layers value, default: all\n");
    printf("  --moe-vram <n>       server --q4e-moe-vram value, default: 10\n");
    printf("  --warm-min-freq <n>  server --q4e-warm-min-freq for the ON phase, default: 1 (legacy, warm all)\n");
    printf("  --oracle <path>      optional llama-server exe for OFF phase\n");
    printf("                       (unset: OFF = q4e-server + --no-q4e-prefetch, same placement)\n");
    printf("  --oracle-ncmoe <n>   oracle -ncmoe value, default: 38\n");
    printf("  --health-timeout <s> health poll deadline in seconds, default: 480\n");
    printf("  --log-dir <dir>      evidence dir, created if missing, default: q4e-bench-run\n");
    printf("  --phases <set>       both|on|off, default: both\n");
    printf("  --v6                 run the v6 component micro-benchmark and exit\n");
    printf("  --v6-kv100k          with --v6: kv section = 300-step sliding window over 100k ctx\n");
    printf("  --help               show this help\n");
    printf("\n");
    printf("notes:\n");
    printf("  - phases are strictly serial: ON is killed and confirmed gone before OFF starts\n");
    printf("  - kills only the child PIDs it spawned; never touches any other process\n");
    printf("  - tps = usage.completion_tokens / request wall time (temp=0, fixed prompt)\n");
    printf("    (this fork has no top-level time_total; curl %%{time_total} semantics)\n");
    printf("  - exit 0 + Q4E_BENCH_DONE verdict=<PASS|FAIL|INCOMPLETE> on mechanical success\n");
    printf("    (a FAIL verdict is a perf result in bench_report.json, not a tool error)\n");
    printf("  - exit 1 + Q4E_BENCH_FAIL <phase>: <reason> on mechanical failure\n");
}

static std::string base_name(const std::string & p) {
    const size_t i = p.find_last_of("/\\");
    return i == std::string::npos ? p : p.substr(i + 1);
}

#ifdef _WIN32

static std::wstring to_wide(const std::string & s) {
    if (s.empty()) {
        return std::wstring();
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), nullptr, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), w.data(), n);
    return w;
}

static std::string to_utf8(const std::wstring & w) {
    if (w.empty()) {
        return std::string();
    }
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int) w.size(), nullptr, 0, nullptr, nullptr);
    std::string s((size_t) n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int) w.size(), &s[0], n, nullptr, nullptr);
    return s;
}

// MSVC argv is CP_ACP-encoded; decode back to UTF-8 so non-ASCII paths stay intact
static std::string ansi_arg_to_utf8(const char * ansi) {
    if (!ansi) {
        return std::string();
    }
    const int nw = MultiByteToWideChar(CP_ACP, 0, ansi, -1, nullptr, 0);
    if (nw <= 0) {
        return std::string();
    }
    std::wstring w((size_t) nw - 1, L'\0');
    MultiByteToWideChar(CP_ACP, 0, ansi, -1, &w[0], nw - 1);
    return to_utf8(w);
}

#endif

// MSVC narrow file APIs misread UTF-8 paths, so use the wide API on Windows
static bool make_dirs(const std::string & path) {
#ifdef _WIN32
    const std::wstring w = to_wide(path);
    std::wstring acc;
    for (size_t i = 0; i < w.size(); ++i) {
        acc.push_back(w[i]);
        if (w[i] == L'\\' || w[i] == L'/' || i + 1 == w.size()) {
            if (!(acc.size() == 2 && acc[1] == L':')) {
                CreateDirectoryW(acc.c_str(), nullptr);
            }
        }
    }
    return GetFileAttributesW(w.c_str()) != INVALID_FILE_ATTRIBUTES;
#else
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    return !ec;
#endif
}

static bool save_file(const std::string & path, const std::string & content) {
#ifdef _WIN32
    const std::wstring w = to_wide(path);
    HANDLE h = CreateFileW(w.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }
    DWORD written = 0;
    const bool ok = WriteFile(h, content.data(), (DWORD) content.size(), &written, nullptr)
                    && written == (DWORD) content.size();
    CloseHandle(h);
    return ok;
#else
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    f << content;
    return (bool) f;
#endif
}

static bool read_file(const std::string & path, std::string & out) {
#ifdef _WIN32
    const std::wstring w = to_wide(path);
    HANDLE h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(h, &sz) || sz.QuadPart < 0) {
        CloseHandle(h);
        return false;
    }
    out.assign((size_t) sz.QuadPart, '\0');
    DWORD rd = 0;
    const bool ok = sz.QuadPart == 0
                    || (ReadFile(h, &out[0], (DWORD) sz.QuadPart, &rd, nullptr) && rd == (DWORD) sz.QuadPart);
    CloseHandle(h);
    return ok;
#else
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(f), {});
    return true;
#endif
}

static void print_tail(const std::string & path, size_t n) {
    std::string content;
    if (!read_file(path, content)) {
        return;
    }
    std::istringstream f(content);
    if (!f) {
        return;
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) {
        lines.push_back(line);
    }
    const size_t start = lines.size() > n ? lines.size() - n : 0;
    for (size_t i = start; i < lines.size(); ++i) {
        printf("%s\n", lines[i].c_str());
    }
}

#ifdef _WIN32

static bool snapshot_scan(bool by_name, const std::wstring & name, DWORD pid) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) {
        return false;
    }
    bool found = false;
    PROCESSENTRY32W pe{};
    pe.dwSize = sizeof(pe);
    for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe)) {
        if (by_name ? _wcsicmp(pe.szExeFile, name.c_str()) == 0 : pe.th32ProcessID == pid) {
            found = true;
            break;
        }
    }
    CloseHandle(snap);
    return found;
}

static bool exe_running(const std::string & exe_name) {
    return snapshot_scan(true, to_wide(exe_name), 0);
}

static bool pid_present(DWORD pid) {
    return snapshot_scan(false, std::wstring(), pid);
}

#else

static bool port_in_use(const std::string & host, int port) {
    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo * res = nullptr;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0) {
        return false;
    }
    bool busy = false;
    const int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd >= 0) {
        struct timeval tv{};
        tv.tv_sec = 2;
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        if (connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
            busy = true;
        }
        ::close(fd);
    }
    freeaddrinfo(res);
    return busy;
}

#endif

static bool port_in_use(const std::string & host, int port) {
#ifdef _WIN32
    httplib::Client cli(host, port);
    cli.set_connection_timeout(2);
    cli.set_read_timeout(2);
    return cli.Get("/") != nullptr;
#else
    return port_in_use(host, port);
#endif
}

static bool child_alive(const child & ch) {
    if (!ch.started) {
        return false;
    }
#ifdef _WIN32
    DWORD code = 0;
    return GetExitCodeProcess(ch.process, &code) && code == STILL_ACTIVE;
#else
    int st = 0;
    return waitpid(ch.pid, &st, WNOHANG) == 0;
#endif
}

// kill the child (only the PID we spawned) and wait up to 180s for exit
static double child_kill(child & ch) {
    if (!ch.started) {
        return 0.0;
    }
    ch.started = false;
    const auto t0 = std::chrono::steady_clock::now();
#ifdef _WIN32
    TerminateProcess(ch.process, 1);
    // the exit code is set before a huge process is fully released, wait for the PID to go
    while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(180) && pid_present(ch.pid)) {
        Sleep(500);
    }
    CloseHandle(ch.process);
#else
    kill(ch.pid, SIGKILL);
    int st = 0;
    while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(180)) {
        if (waitpid(ch.pid, &st, WNOHANG) != 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
#endif
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static bool child_start(const std::string & exe, const std::vector<std::string> & args,
                        const std::string & out_log, const std::string & err_log,
                        child & ch, std::string & err) {
#ifdef _WIN32
    const auto q = [](const std::string & s) { return std::string("\"") + s + "\""; };
    std::string cmd = q(exe);
    for (const std::string & a : args) {
        cmd += " " + q(a);
    }
    const auto open_log = [](const std::string & path) -> HANDLE {
        return CreateFileW(to_wide(path).c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    };
    const HANDLE out = open_log(out_log);
    const HANDLE errf = open_log(err_log);
    if (out == INVALID_HANDLE_VALUE || errf == INVALID_HANDLE_VALUE) {
        if (out != INVALID_HANDLE_VALUE) {
            CloseHandle(out);
        }
        if (errf != INVALID_HANDLE_VALUE) {
            CloseHandle(errf);
        }
        err = "CreateFileW failed for log files (error " + std::to_string((int) GetLastError()) + ")";
        return false;
    }
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = out;
    si.hStdError = errf;
    PROCESS_INFORMATION pi{};
    const std::wstring wcmd = to_wide(cmd);
    if (!CreateProcessW(nullptr, const_cast<wchar_t *>(wcmd.c_str()), nullptr, nullptr, TRUE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        const int code = (int) GetLastError();
        CloseHandle(out);
        CloseHandle(errf);
        err = "CreateProcessW failed (error " + std::to_string(code) + ")";
        return false;
    }
    CloseHandle(pi.hThread);
    CloseHandle(out);
    CloseHandle(errf);
    ch.process = pi.hProcess;
    ch.pid = pi.dwProcessId;
    ch.started = true;
    return true;
#else
    const int out = open(out_log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    const int errf = open(err_log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out < 0 || errf < 0) {
        if (out >= 0) {
            ::close(out);
        }
        if (errf >= 0) {
            ::close(errf);
        }
        err = "failed to open log files";
        return false;
    }
    const pid_t pid = fork();
    if (pid < 0) {
        ::close(out);
        ::close(errf);
        err = "fork failed";
        return false;
    }
    if (pid == 0) {
        dup2(out, 1);
        dup2(errf, 2);
        std::vector<std::string> a = { exe };
        a.insert(a.end(), args.begin(), args.end());
        std::vector<char *> av;
        for (std::string & s : a) {
            av.push_back(&s[0]);
        }
        av.push_back(nullptr);
        execv(exe.c_str(), av.data());
        _exit(127);
    }
    ::close(out);
    ::close(errf);
    ch.pid = pid;
    ch.started = true;
    return true;
#endif
}

static http_result post_chat(const conf & c, int port, int max_tokens) {
    http_result r;
    httplib::Client cli(c.host, port);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(900);
    nlohmann::json j;
    j["messages"] = nlohmann::json::array();
    nlohmann::json msg;
    msg["role"] = "user";
    msg["content"] = c.prompt;
    j["messages"].push_back(std::move(msg));
    j["max_tokens"] = max_tokens;
    j["temperature"] = 0;
    j["stream"] = false;
    const httplib::Headers headers{{"Authorization", "Bearer " + c.api_key}};
    const auto res = cli.Post("/v1/chat/completions", headers, j.dump(), "application/json");
    if (!res) {
        return r;
    }
    r.ok = true;
    r.status = res->status;
    r.body = res->body;
    return r;
}

static http_result get_auth(const conf & c, int port, const std::string & path) {
    http_result r;
    httplib::Client cli(c.host, port);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(30);
    const httplib::Headers headers{{"Authorization", "Bearer " + c.api_key}};
    const auto res = cli.Get(path, headers);
    if (!res) {
        return r;
    }
    r.ok = true;
    r.status = res->status;
    r.body = res->body;
    return r;
}

// one full phase: G0 -> spawn -> health -> warmup -> measured -> stats -> kill
static std::string run_phase(const conf & c, const char * tag, int port, bool use_oracle,
                             bool prefetch, phase_result & res) {
    res.ran = true;
    res.prefetch = prefetch;
    const std::string exe = use_oracle ? c.oracle : c.server;
    res.server = exe;

    child ch;
    const auto fail = [&](const std::string & reason) -> std::string {
        if (ch.started) {
            const double waited = child_kill(ch);
            printf("[%s] cleanup: killed own child pid=%d, waited %.1fs\n", tag, (int) ch.pid, waited);
        }
        return reason;
    };

    // G0: no residual process of this exe (Windows), port free
#ifdef _WIN32
    if (exe_running(base_name(exe))) {
        return fail("residual process still running: " + base_name(exe));
    }
#endif
    if (port_in_use(c.host, port)) {
        return fail("port " + std::to_string(port) + " already in use");
    }
    printf("[%s] G0 PASS: no residual %s; port %d free\n", tag, base_name(exe).c_str(), port);

    // START
    std::vector<std::string> args;
    if (use_oracle) {
        args = {"-m", c.model, "--host", c.host, "--port", std::to_string(port),
                "-ngl", "all", "-ncmoe", std::to_string(c.oracle_ncmoe),
                "--ctx-size", std::to_string(c.ctx_size)};
    } else {
        args = {"--model", c.model, "--host", c.host, "--port", std::to_string(port),
                "--api-key", c.api_key,
                "--n-gpu-layers", c.gpu_layers,
                "--q4e-moe-vram", std::to_string(c.moe_vram),
                "--q4e-warm-min-freq", std::to_string(c.warm_min_freq),
                "--ctx-size", std::to_string(c.ctx_size),
                "--parallel", "1"};
        if (!prefetch) {
            args.push_back("--no-q4e-prefetch");
        }
    }
    const std::string out_log = c.log_dir + "/" + tag + "_server.out.log";
    const std::string err_log = c.log_dir + "/" + tag + "_server.err.log";
    std::string spawn_err;
    if (!child_start(exe, args, out_log, err_log, ch, spawn_err)) {
        return fail("spawn failed: " + spawn_err);
    }
    printf("[%s] START pid=%d\n", tag, (int) ch.pid);

    // HEALTH: /health 200 within deadline; fail if the child exits first
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(c.health_timeout);
        bool ok = false;
        while (std::chrono::steady_clock::now() < deadline) {
            httplib::Client cli(c.host, port);
            cli.set_connection_timeout(5);
            cli.set_read_timeout(5);
            const auto res = cli.Get("/health");
            if (res && res->status == 200) {
                ok = true;
                break;
            }
            if (!child_alive(ch)) {
                printf("[%s] FAIL: server process exited before /health 200\n", tag);
                printf("--- tail(30) %s ---\n", err_log.c_str());
                print_tail(err_log, 30);
                return fail("server exited before /health 200");
            }
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
        if (!ok) {
            printf("[%s] FAIL: /health never returned 200 within %ds\n", tag, c.health_timeout);
            printf("--- tail(30) %s ---\n", err_log.c_str());
            print_tail(err_log, 30);
            return fail("health timeout");
        }
        printf("[%s] HEALTH PASS: /health 200\n", tag);
    }

    // WARMUP: fixed prompt, temp=0, max_tokens=warmup-tokens
    {
        const http_result r = post_chat(c, port, c.warmup_tokens);
        if (!r.ok) {
            return fail("warmup completion: no response");
        }
        if (!save_file(c.log_dir + "/" + tag + "_warmup.json", r.body)) {
            return fail("failed to save " + std::string(tag) + "_warmup.json");
        }
        printf("[%s] WARMUP done (status=%d)\n", tag, r.status);
    }

    // MEASURE: same prompt, max_tokens=measure-tokens; tps = completion_tokens / time_total
    {
        // client-side wall clock of the whole request (curl %{time_total} semantics)
        const auto t0 = std::chrono::steady_clock::now();
        const http_result r = post_chat(c, port, c.measure_tokens);
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!r.ok) {
            return fail("measured completion: no response");
        }
        if (r.status != 200) {
            return fail("measured completion: expected 200, got " + std::to_string(r.status));
        }
        if (!save_file(c.log_dir + "/" + tag + "_measure.json", r.body)) {
            return fail("failed to save " + std::string(tag) + "_measure.json");
        }
        const auto j = nlohmann::json::parse(r.body, nullptr, false);
        if (j.is_discarded()) {
            return fail("measured response is not valid JSON");
        }
        long long tokens = 0;
        if (j.contains("usage") && j["usage"].contains("completion_tokens")
            && j["usage"]["completion_tokens"].is_number()) {
            tokens = j["usage"]["completion_tokens"].get<long long>();
        }
        if (tokens < 1) {
            return fail("completion_tokens < 1 (got " + std::to_string(tokens) + ")");
        }
        double tt = 0.0;
        // this fork reports "timings", not a top-level time_total; wall clock is the reference
        if (j.contains("time_total") && j["time_total"].is_number()) {
            tt = j["time_total"].get<double>();
        }
        if (tt <= 0.0) {
            tt = wall;
        }
        if (tt <= 0.0) {
            return fail("time_total and wall clock are both missing or <= 0");
        }
        res.completion_tokens = tokens;
        res.time_total = tt;
        res.tps = (double) tokens / tt;
        printf("[%s] MEASURE tps=%.3f (tokens=%lld time_total=%.3f)\n", tag, res.tps, tokens, tt);
    }

    // STATS: q4e only (the oracle has no /q4e/stats)
    if (!use_oracle) {
        const http_result r = get_auth(c, port, "/q4e/stats");
        if (!r.ok) {
            return fail("GET /q4e/stats: no response");
        }
        if (r.status != 200) {
            return fail("GET /q4e/stats: expected 200, got " + std::to_string(r.status));
        }
        if (!save_file(c.log_dir + "/" + tag + "_stats.json", r.body)) {
            return fail("failed to save " + std::string(tag) + "_stats.json");
        }
        const auto j = nlohmann::json::parse(r.body, nullptr, false);
        if (j.is_discarded()) {
            return fail(std::string(tag) + "_stats.json is not valid JSON");
        }
        nlohmann::json st;
        const char * keys[] = {"q4e_prefetch", "prefetch_overlap_ratio", "prefetch_io_issued",
                               "prefetch_overlap_hits", "touches", "layers_touched",
                               "hits_vram", "hits_ram", "hits_ssd"};
        for (const char * k : keys) {
            st[k] = j.contains(k) ? j[k] : nlohmann::json(nullptr);
        }
        res.stats = std::move(st);
        res.has_stats = true;
        if (!prefetch) {
            // OFF phase expectation (not a mechanical gate)
            const bool num = res.stats["q4e_prefetch"].is_number();
            if (num && res.stats["q4e_prefetch"].get<long long>() != 0) {
                printf("[%s] NOTE: q4e_prefetch=%s (expected 0)\n", tag,
                       res.stats["q4e_prefetch"].dump().c_str());
            }
        }
        printf("[%s] STATS saved (q4e_prefetch=%s overlap_ratio=%s touches=%s)\n", tag,
               res.stats["q4e_prefetch"].dump().c_str(),
               res.stats["prefetch_overlap_ratio"].dump().c_str(),
               res.stats["touches"].dump().c_str());
    }

    // KILL: own PID only, wait up to 180s, confirm gone
    {
        const double waited = child_kill(ch);
#ifdef _WIN32
        if (pid_present(ch.pid)) {
            return "own pid " + std::to_string((int) ch.pid) + " still present after kill";
        }
#endif
        printf("[%s] KILL PASS: own child pid=%d gone after %.1fs\n", tag, (int) ch.pid, waited);
    }
    return "";
}

int main(int argc, char ** argv) {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) {
#ifdef _WIN32
        args.push_back(ansi_arg_to_utf8(argv[i]));
#else
        args.push_back(argv[i]);
#endif
    }
    if (args.empty()) {
        return 2;
    }
    conf c;
    bool show_help = false;
    try {
        for (size_t i = 1; i < args.size(); ++i) {
            const std::string a = args[i];
            const auto value = [&](const char * name) -> const std::string & {
                if (i + 1 >= args.size()) {
                    fprintf(stderr, "error: missing value for %s\n", name);
                    std::exit(2);
                }
                return args[++i];
            };
            if (a == "--help" || a == "-h") {
                show_help = true;
            } else if (a == "--server") {
                c.server = value("--server");
            } else if (a == "--model") {
                c.model = value("--model");
            } else if (a == "--host") {
                c.host = value("--host");
            } else if (a == "--port-on") {
                c.port_on = std::stoi(value("--port-on"));
            } else if (a == "--port-off") {
                c.port_off = std::stoi(value("--port-off"));
            } else if (a == "--api-key") {
                c.api_key = value("--api-key");
            } else if (a == "--ctx-size") {
                c.ctx_size = std::stoi(value("--ctx-size"));
            } else if (a == "--warmup-tokens") {
                c.warmup_tokens = std::stoi(value("--warmup-tokens"));
            } else if (a == "--measure-tokens") {
                c.measure_tokens = std::stoi(value("--measure-tokens"));
            } else if (a == "--prompt") {
                c.prompt = value("--prompt");
            } else if (a == "--gpu-layers") {
                c.gpu_layers = value("--gpu-layers");
            } else if (a == "--moe-vram") {
                c.moe_vram = std::stoi(value("--moe-vram"));
            } else if (a == "--warm-min-freq") {
                c.warm_min_freq = std::stoi(value("--warm-min-freq"));
            } else if (a == "--oracle") {
                c.oracle = value("--oracle");
            } else if (a == "--oracle-ncmoe") {
                c.oracle_ncmoe = std::stoi(value("--oracle-ncmoe"));
            } else if (a == "--health-timeout") {
                c.health_timeout = std::stoi(value("--health-timeout"));
            } else if (a == "--log-dir") {
                c.log_dir = value("--log-dir");
            } else if (a == "--phases") {
                c.phases = value("--phases");
            } else if (a == "--v6") {
                c.v6 = true;
            } else if (a == "--v6-kv100k") {
                c.v6_kv100k = true;
            } else {
                fprintf(stderr, "error: unknown arg %s (see --help)\n", a.c_str());
                return 2;
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: bad arg value: %s\n", e.what());
        return 2;
    }

    if (show_help) {
        usage();
        return 0;
    }
    if (c.v6) {
        q4e_bench_v6_opts opt;
        opt.kv100k = c.v6_kv100k;
        const q4e_bench_v6_result r = q4e_bench_v6_run(opt);
        printf("%s", q4e_bench_v6_text(r).c_str());
        return r.ok ? 0 : 1;
    }
    if (c.server.empty() || c.model.empty()) {
        fprintf(stderr, "error: --server and --model are required (see --help)\n");
        return 2;
    }
    if (c.phases != "both" && c.phases != "on" && c.phases != "off") {
        fprintf(stderr, "error: --phases must be both|on|off (got %s)\n", c.phases.c_str());
        return 2;
    }

    if (!make_dirs(c.log_dir)) {
        fprintf(stderr, "error: cannot create log dir %s\n", c.log_dir.c_str());
        return 2;
    }

    const bool do_on = c.phases != "off";
    const bool do_off = c.phases != "on";

    // strictly serial: OFF only starts after ON is killed and confirmed gone
    phase_result on_res, off_res;
    if (do_on) {
        const std::string err = run_phase(c, "on", c.port_on, false, true, on_res);
        if (!err.empty()) {
            printf("Q4E_BENCH_FAIL on: %s\n", err.c_str());
            return 1;
        }
    }
    if (do_off) {
        const std::string err = run_phase(c, "off", c.port_off, !c.oracle.empty(), false, off_res);
        if (!err.empty()) {
            printf("Q4E_BENCH_FAIL off: %s\n", err.c_str());
            return 1;
        }
    }

    // report
    nlohmann::json rep;
    rep["model"] = c.model;
    rep["host"] = c.host;
    rep["prompt"] = c.prompt;
    rep["temp"] = 0;
    rep["warmup_tokens"] = c.warmup_tokens;
    rep["measure_tokens"] = c.measure_tokens;
    const auto phase_json = [](const phase_result & r) -> nlohmann::json {
        nlohmann::json j;
        j["server"] = r.server;
        j["prefetch"] = r.prefetch;
        j["completion_tokens"] = r.completion_tokens;
        j["time_total"] = r.time_total;
        j["tps"] = r.tps;
        j["stats"] = r.has_stats ? r.stats : nlohmann::json(nullptr);
        return j;
    };
    rep["on"] = on_res.ran ? phase_json(on_res) : nlohmann::json(nullptr);
    rep["off"] = off_res.ran ? phase_json(off_res) : nlohmann::json(nullptr);

    const bool incomplete = !on_res.ran || !off_res.ran || !on_res.has_stats;
    bool g_on_ge_off = false, g_195 = false, g_overlap = false;
    if (!incomplete) {
        g_on_ge_off = on_res.tps >= off_res.tps;
        g_195 = on_res.tps >= 1.95;
        const auto & ratio = on_res.stats["prefetch_overlap_ratio"];
        g_overlap = ratio.is_number() && ratio.get<double>() >= 0.80;
    }
    std::string verdict = incomplete ? "INCOMPLETE"
                        : (g_on_ge_off && g_195 && g_overlap) ? "PASS" : "FAIL";
    nlohmann::json gates;
    gates["on_tps_ge_off"] = g_on_ge_off;
    gates["on_tps_ge_1_95"] = g_195;
    gates["on_tps_ge_off_and_1_95"] = g_on_ge_off && g_195;
    gates["overlap_ge_0_80"] = g_overlap;
    gates["verdict"] = verdict;
    rep["gates"] = gates;

    const std::string report_path = c.log_dir + "/bench_report.json";
    if (!save_file(report_path, rep.dump(2) + "\n")) {
        printf("Q4E_BENCH_FAIL report: failed to save bench_report.json\n");
        return 1;
    }

    printf("\n== q4e-bench summary ==\n");
    printf("model    : %s\n", c.model.c_str());
    printf("prompt   : %s (temp=0, warmup=%d, measure=%d)\n", c.prompt.c_str(), c.warmup_tokens, c.measure_tokens);
    if (on_res.ran) {
        printf("ON       : server=%s prefetch=on  tps=%.3f tokens=%lld time_total=%.3f\n",
               on_res.server.c_str(), on_res.tps, on_res.completion_tokens, on_res.time_total);
    }
    if (off_res.ran) {
        printf("OFF      : server=%s prefetch=off tps=%.3f tokens=%lld time_total=%.3f\n",
               off_res.server.c_str(), off_res.tps, off_res.completion_tokens, off_res.time_total);
    }
    if (on_res.has_stats) {
        const auto & st = on_res.stats;
        printf("overlap  : %s  (io_issued=%s touches=%s layers_touched=%s)\n",
               st["prefetch_overlap_ratio"].dump().c_str(), st["prefetch_io_issued"].dump().c_str(),
               st["touches"].dump().c_str(), st["layers_touched"].dump().c_str());
    }
    printf("gates    : on_tps_ge_off=%s on_tps_ge_1_95=%s overlap_ge_0_80=%s\n",
           g_on_ge_off ? "true" : "false", g_195 ? "true" : "false", g_overlap ? "true" : "false");
    printf("verdict  : %s\n", verdict.c_str());
    printf("report   : %s\n", report_path.c_str());
    printf("Q4E_BENCH_DONE verdict=%s\n", verdict.c_str());
    return 0;
}
