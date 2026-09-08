// q4e-ngram: ngram prefetch diag/accept client for q4e-server
// replaces legacy_v1/tools/ngram_diag.py (4-stage A/B/C/D) and ngram_accept.py (P1-P4 drop gate)

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
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
    #include <winsock2.h>
    #include <windows.h>
    #include <tlhelp32.h>
    #include <string.h>
    #ifndef SO_CONNECTTIMEO
        #define SO_CONNECTTIMEO 0x1200
    #endif
#else
    #include <errno.h>
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
    std::string mode;
    std::string server;
    std::string model;
    std::string host = "127.0.0.1";
    int         port = 8132;
    std::string api_key = "bkey";
    int         ctx_size = 2048;
    int         warmup_tokens = 300;
    int         tokens = 300;
    int         tail = 150;
    std::string prompt = "List prime numbers starting from 2, one number per line, no explanations, keep going.";
    std::string gpu_layers = "all";
    int         moe_vram = 10;
    int         health_timeout = 480;
    std::string log_dir = "q4e-ngram-run";
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
    std::string name;
    std::string mode;
    int         n_tokens = 0;
    double      cum_tps = 0.0;
    double      tail_tps = 0.0;
    nlohmann::json rolling50 = nlohmann::json::array();
    double      rows_warmed = -1.0;
};

struct stream_result {
    bool        ok = false;
    int         status = 0;
    std::string body;
    std::vector<double> stamps;
    double      end_ts = 0.0;
    nlohmann::json rolling50 = nlohmann::json::array();
};

static void usage() {
    printf("q4e-ngram: ngram prefetch diag/accept client for q4e-server\n");
    printf("(replaces legacy ngram_diag.py 4-stage A/B/C/D and ngram_accept.py P1-P4 drop gate)\n");
    printf("\n");
    printf("usage: q4e-ngram <diag|accept> --server <q4e-server-exe> --model <gguf> [options]\n");
    printf("\n");
    printf("modes:\n");
    printf("  diag    4-stage attribution: A=off B=shim C=full D=full (rolling-50 series)\n");
    printf("  accept  drop gate: P1=off P2=full P3=off P4=full (cum + tail, drop < 1%% gate)\n");
    printf("\n");
    printf("required:\n");
    printf("  --server <path>      path to q4e-server binary\n");
    printf("  --model <path>       path to GGUF model file\n");
    printf("\n");
    printf("options:\n");
    printf("  --host <ip>          default: 127.0.0.1\n");
    printf("  --port <n>           default: 8132\n");
    printf("  --api-key <key>      bearer key for server + requests, default: bkey\n");
    printf("  --ctx-size <n>       server --ctx-size, default: 2048\n");
    printf("  --warmup-tokens <n>  warmup completion max_tokens, default: 300\n");
    printf("  --tokens <n>         per-phase max_tokens, default: 300\n");
    printf("  --tail <n>           accept tail window (last N tokens), default: 150\n");
    printf("  --prompt <text>      default: List prime numbers starting from 2, one number per line, no explanations, keep going.\n");
    printf("  --gpu-layers <v>     server --n-gpu-layers value, default: all\n");
    printf("  --moe-vram <n>       server --q4e-moe-vram value, default: 10\n");
    printf("  --health-timeout <s> health poll deadline in seconds, default: 480\n");
    printf("  --log-dir <dir>      evidence dir, created if missing, default: q4e-ngram-run\n");
    printf("  --help               show this help\n");
    printf("\n");
    printf("notes:\n");
    printf("  - all phases run strictly serial on ONE server instance;\n");
    printf("    the ngram mode is switched per phase via POST /q4e/ngram\n");
    printf("  - kills only the child PID it spawned; never touches any other process\n");
    printf("  - timed phases = SSE streaming with per-chunk client timestamps (temp=0, fixed prompt)\n");
    printf("  - accept PASS = max of the 4 drops (cum + tail, P1/P2 and P3/P4) < 1.0%%\n");
    printf("  - exit 0 + Q4E_NGRAM_DONE verdict=<PASS|FAIL|n/a> on mechanical success\n");
    printf("    (a FAIL verdict is perf data in the report, not a tool error)\n");
    printf("  - exit 1 + Q4E_NGRAM_FAIL <phase>: <reason> on mechanical failure\n");
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

// trim leading/trailing whitespace (space, tab, CR)
static std::string trim(const std::string & s) {
    const char * ws = " \t\r";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) {
        return "";
    }
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
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

static bool ws_started = false;
static bool ws_init() {
    if (ws_started) {
        return true;
    }
    WSADATA d;
    if (WSAStartup(MAKEWORD(2, 2), &d) != 0) {
        return false;
    }
    ws_started = true;
    return true;
}
static void ws_fini() {
    if (ws_started) {
        WSACleanup();
        ws_started = false;
    }
}

#else

static void ws_fini() {
}

static bool port_in_use_raw(const std::string & host, int port) {
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
    return port_in_use_raw(host, port);
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
        dup2(err, 2);
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

static std::string set_ngram_mode(const conf & c, const char * tag, const char * mode) {
    httplib::Client cli(c.host, c.port);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(30);
    nlohmann::json j;
    j["mode"] = mode;
    const httplib::Headers headers{{"Authorization", "Bearer " + c.api_key}};
    const auto res = cli.Post("/q4e/ngram", headers, j.dump(), "application/json");
    if (!res) {
        return "POST /q4e/ngram: no response";
    }
    if (!save_file(c.log_dir + "/" + std::string(tag) + "_mode.json", res->body)) {
        return "failed to save " + std::string(tag) + "_mode.json";
    }
    if (res->status != 200) {
        return "POST /q4e/ngram: expected 200, got " + std::to_string(res->status);
    }
    const auto parsed = nlohmann::json::parse(res->body, nullptr, false);
    if (parsed.is_discarded() || !parsed.contains("ok") || !parsed["ok"].is_boolean()
        || !parsed["ok"].get<bool>()) {
        return "POST /q4e/ngram: ok != true; body=" + res->body.substr(0, 200);
    }
    return "";
}

static std::string get_stats(const conf & c, const std::string & tag, nlohmann::json & out) {
    httplib::Client cli(c.host, c.port);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(30);
    const httplib::Headers headers{{"Authorization", "Bearer " + c.api_key}};
    const auto res = cli.Get("/q4e/stats", headers);
    if (!res) {
        return "GET /q4e/stats: no response";
    }
    if (res->status != 200) {
        return "GET /q4e/stats: expected 200, got " + std::to_string(res->status);
    }
    if (!save_file(c.log_dir + "/" + tag + "_stats.json", res->body)) {
        return "failed to save " + tag + "_stats.json";
    }
    out = nlohmann::json::parse(res->body, nullptr, false);
    if (out.is_discarded()) {
        return tag + "_stats.json is not valid JSON";
    }
    return "";
}

static std::string do_warmup(const conf & c) {
    nlohmann::json j;
    j["prompt"] = c.prompt;
    j["max_tokens"] = c.warmup_tokens;
    j["temperature"] = 0;
    j["stream"] = false;
    httplib::Client cli(c.host, c.port);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(1800);
    const httplib::Headers headers{{"Authorization", "Bearer " + c.api_key}};
    const auto res = cli.Post("/v1/completions", headers, j.dump(), "application/json");
    if (!res) {
        return "warmup: no response";
    }
    if (!save_file(c.log_dir + "/warmup.json", res->body)) {
        return "failed to save warmup.json";
    }
    if (res->status != 200) {
        return "warmup: expected 200, got " + std::to_string(res->status);
    }
    const auto parsed = nlohmann::json::parse(res->body, nullptr, false);
    long long tokens = 0;
    if (!parsed.is_discarded() && parsed.contains("usage") && parsed["usage"].contains("completion_tokens")
        && parsed["usage"]["completion_tokens"].is_number()) {
        tokens = parsed["usage"]["completion_tokens"].get<long long>();
    }
    if (tokens < 1) {
        return "warmup: completion_tokens < 1 (got " + std::to_string(tokens) + ")";
    }
    printf("WARMUP done (%lld tokens, status=%d)\n", tokens, res->status);
    return "";
}

// one timed phase: SSE streaming with per-chunk client timestamps (httplib cannot stream)
static std::string stream_phase(const conf & c, const char * tag, bool with_rolling, stream_result & res) {
#ifdef _WIN32
    if (!ws_init()) {
        return "WSAStartup failed";
    }
#endif
    nlohmann::json body_j;
    body_j["prompt"] = c.prompt;
    body_j["max_tokens"] = c.tokens;
    body_j["temperature"] = 0;
    body_j["stream"] = true;
    const std::string body = body_j.dump();
    std::string req;
    req += "POST /v1/completions HTTP/1.1\r\n";
    req += "Host: " + c.host + "\r\n";
    req += "Authorization: Bearer " + c.api_key + "\r\n";
    req += "Content-Type: application/json\r\n";
    req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    req += "Connection: close\r\n\r\n";
    req += body;

    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo * ai = nullptr;
    if (getaddrinfo(c.host.c_str(), std::to_string(c.port).c_str(), &hints, &ai) != 0) {
        return "getaddrinfo failed for " + c.host + ":" + std::to_string(c.port);
    }
#ifdef _WIN32
    SOCKET sock = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (sock == INVALID_SOCKET) {
        freeaddrinfo(ai);
        return "socket() failed (error " + std::to_string(WSAGetLastError()) + ")";
    }
    const DWORD cto = 10000;
    setsockopt(sock, SOL_SOCKET, SO_CONNECTTIMEO, (const char *) &cto, sizeof(cto));
    if (connect(sock, ai->ai_addr, ai->ai_addrlen) == SOCKET_ERROR) {
        freeaddrinfo(ai);
        closesocket(sock);
        return "connect failed (error " + std::to_string(WSAGetLastError()) + ")";
    }
    const DWORD rto = 120000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char *) &rto, sizeof(rto));
#else
    int sock = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (sock < 0) {
        freeaddrinfo(ai);
        return "socket() failed";
    }
    if (connect(sock, ai->ai_addr, ai->ai_addrlen) != 0) {
        freeaddrinfo(ai);
        ::close(sock);
        return "connect failed";
    }
    struct timeval tv{};
    tv.tv_sec = 120;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
    freeaddrinfo(ai);

    std::string err;
    size_t sent = 0;
    while (err.empty() && sent < req.size()) {
#ifdef _WIN32
        const int n = send(sock, req.data() + sent, (int) (req.size() - sent), 0);
        if (n == SOCKET_ERROR) {
            err = "send failed (error " + std::to_string(WSAGetLastError()) + ")";
#else
        const int n = (int) send(sock, req.data() + sent, req.size() - sent, 0);
        if (n < 0) {
            err = "send failed";
#endif
        } else {
            sent += (size_t) n;
        }
    }

    if (err.empty()) {
        const bool dbg = (getenv("Q4E_NGRAM_DEBUG") != nullptr);
        const auto t0 = std::chrono::steady_clock::now();
        char buf[8192];
        std::string netbuf;
        bool stream_done = false;
        double last_roll = 0.0;
        auto handle_event = [&](const std::string & ev) -> bool {
            size_t p = 0;
            while (p < ev.size()) {
                const size_t eol = ev.find('\n', p);
                std::string line = ev.substr(p, eol == std::string::npos ? std::string::npos : eol - p);
                line = trim(line);
                if (line.rfind("data:", 0) == 0) {
                    const std::string payload = trim(line.substr(5));
                    if (payload == "[DONE]") {
                        return true;
                    }
                    // one arrival per data: chunk (legacy per-chunk stamping)
                    const double ts = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    res.stamps.push_back(ts);
                    const auto chunk = nlohmann::json::parse(payload, nullptr, false);
                    if (!chunk.is_discarded()) {
                        (void) chunk;
                    }
                    if (with_rolling && (res.stamps.size() % 50) == 0) {
                        const double span = ts - last_roll;
                        nlohmann::json e;
                        e["tok"] = (int) res.stamps.size();
                        e["rolling_tps"] = span > 0.0 ? 50.0 / span : 0.0;
                        e["cum_tps"] = ts > 0.0 ? (double) res.stamps.size() / ts : 0.0;
                        res.rolling50.push_back(std::move(e));
                        last_roll = ts;
                        printf("    [%s] tok %d  rolling %.2f t/s  cum %.2f\n", tag, (int) res.stamps.size(),
                               span > 0.0 ? 50.0 / span : 0.0, ts > 0.0 ? (double) res.stamps.size() / ts : 0.0);
                    }
                }
                if (eol == std::string::npos) {
                    break;
                }
                p = eol + 1;
            }
            return false;
        };
        for (;;) {
            int n;
#ifdef _WIN32
            n = recv(sock, buf, sizeof(buf), 0);
            if (n == SOCKET_ERROR) {
                err = "recv failed (error " + std::to_string(WSAGetLastError()) + ")";
                break;
            }
#else
            n = (int) recv(sock, buf, sizeof(buf), 0);
            if (n < 0) {
                err = (errno == EAGAIN || errno == EWOULDBLOCK) ? "recv timeout" : "recv failed";
                break;
            }
#endif
            if (n == 0) {
                if (dbg) {
                    printf("    [dbg] recv EOF after %zu bytes\n", res.body.size());
                }
                break;
            }
            res.body.append(buf, (size_t) n);
            netbuf.append(buf, (size_t) n);
            if (dbg) {
                printf("    [dbg] recv %d bytes (body now %zu)\n", n, res.body.size());
            }
            while (!stream_done) {
                const size_t a = netbuf.find("\n\n");
                const size_t b = netbuf.find("\r\n\r\n");
                size_t d;
                size_t len;
                if (a == std::string::npos && b == std::string::npos) {
                    break;
                }
                if (b != std::string::npos && (a == std::string::npos || b < a)) {
                    d = b;
                    len = 4;
                } else {
                    d = a;
                    len = 2;
                }
                const std::string ev = netbuf.substr(0, d);
                netbuf.erase(0, d + len);
                if (handle_event(ev)) {
                    stream_done = true;
                }
            }
            if (stream_done) {
                break;
            }
        }
        // wall ends at the [DONE] event or at the last chunk, whichever came
        if (stream_done) {
            res.end_ts = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        } else if (!res.stamps.empty()) {
            res.end_ts = res.stamps.back();
        }
    }

#ifdef _WIN32
    closesocket(sock);
#else
    ::close(sock);
#endif

    if (!err.empty() || res.stamps.empty()) {
        // keep the full raw exchange for post-mortem
        save_file(c.log_dir + "/" + std::string(tag) + "_sse_fail_raw.txt", res.body);
        save_file(c.log_dir + "/" + std::string(tag) + "_sse_fail_req.txt", req);
    }
    if (!err.empty()) {
        return err;
    }
    {
        // status line of the raw response (headers arrive before the SSE body)
        const size_t nl = res.body.find('\n');
        const std::string first = res.body.substr(0, nl);
        const size_t sp = first.find(' ');
        if (sp != std::string::npos && sp + 4 <= first.size()) {
            res.status = atoi(first.c_str() + sp + 1);
        }
    }
    if (res.status != 200) {
        return "SSE status " + std::to_string(res.status) + "; raw=" + res.body.substr(0, 200);
    }
    if (res.stamps.empty()) {
        return "SSE stream carried no token chunks; raw=" + res.body.substr(0, 200);
    }
    res.ok = true;
    return "";
}

static std::string run_phase(const conf & c, const char * name, const char * mode, bool accept_mode,
                             phase_result & res) {
    res.name = name;
    res.mode = mode;
    const bool full = (mode == "full");

    const std::string err_mode = set_ngram_mode(c, name, mode);
    if (!err_mode.empty()) {
        return err_mode;
    }
    printf("[%s] MODE set: %s (ok:true)\n", name, mode);

    nlohmann::json stats_before;
    if (accept_mode && full) {
        const std::string e = get_stats(c, std::string(name) + "_before", stats_before);
        if (!e.empty()) {
            printf("[%s] NOTE: pre-phase stats read failed: %s (volume line skipped)\n", name, e.c_str());
        }
    }

    stream_result sr;
    const std::string err_stream = stream_phase(c, name, !accept_mode, sr);
    if (!err_stream.empty()) {
        return err_stream;
    }
    if (!save_file(c.log_dir + "/" + std::string(name) + "_sse.txt", sr.body)) {
        return "failed to save " + std::string(name) + "_sse.txt";
    }

    res.n_tokens = (int) sr.stamps.size();
    res.cum_tps = sr.end_ts > 0.0 ? (double) res.n_tokens / sr.end_ts : 0.0;
    res.rolling50 = sr.rolling50;
    if (accept_mode) {
        if (sr.stamps.size() > (size_t) c.tail) {
            const double span = sr.stamps.back() - sr.stamps[sr.stamps.size() - (size_t) c.tail - 1];
            res.tail_tps = span > 0.0 ? (double) c.tail / span : 0.0;
        } else {
            res.tail_tps = res.cum_tps;
        }
    }

    if (accept_mode && full) {
        nlohmann::json stats_after;
        const std::string e = get_stats(c, std::string(name) + "_after", stats_after);
        if (e.empty() && stats_before.is_object() && stats_before.contains("ngram_rows_warmed")
            && stats_after.contains("ngram_rows_warmed")
            && stats_before["ngram_rows_warmed"].is_number()
            && stats_after["ngram_rows_warmed"].is_number()) {
            const double before = stats_before["ngram_rows_warmed"].get<double>();
            const double after = stats_after["ngram_rows_warmed"].get<double>();
            res.rows_warmed = after - before;
        }
    }

    if (accept_mode) {
        printf("[%s] %s  cum=%.2f t/s  tail%d=%.2f t/s\n", name, mode, res.cum_tps, c.tail, res.tail_tps);
    } else {
        printf("[%s] %s  cum=%.2f t/s (%d chunks)\n", name, mode, res.cum_tps, res.n_tokens);
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
            } else if (a == "--port") {
                c.port = std::stoi(value("--port"));
            } else if (a == "--api-key") {
                c.api_key = value("--api-key");
            } else if (a == "--ctx-size") {
                c.ctx_size = std::stoi(value("--ctx-size"));
            } else if (a == "--warmup-tokens") {
                c.warmup_tokens = std::stoi(value("--warmup-tokens"));
            } else if (a == "--tokens") {
                c.tokens = std::stoi(value("--tokens"));
            } else if (a == "--tail") {
                c.tail = std::stoi(value("--tail"));
            } else if (a == "--prompt") {
                c.prompt = value("--prompt");
            } else if (a == "--gpu-layers") {
                c.gpu_layers = value("--gpu-layers");
            } else if (a == "--moe-vram") {
                c.moe_vram = std::stoi(value("--moe-vram"));
            } else if (a == "--health-timeout") {
                c.health_timeout = std::stoi(value("--health-timeout"));
            } else if (a == "--log-dir") {
                c.log_dir = value("--log-dir");
            } else if (a.rfind("--", 0) != 0) {
                if (!c.mode.empty()) {
                    fprintf(stderr, "error: multiple modes given (use one of diag|accept)\n");
                    return 2;
                }
                c.mode = a;
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
    if (c.mode != "diag" && c.mode != "accept") {
        fprintf(stderr, "error: mode must be diag or accept (see --help)\n");
        return 2;
    }
    if (c.server.empty() || c.model.empty()) {
        fprintf(stderr, "error: --server and --model are required (see --help)\n");
        return 2;
    }
    if (c.warmup_tokens < 1 || c.tokens < 1 || c.tail < 1) {
        fprintf(stderr, "error: --warmup-tokens, --tokens and --tail must be >= 1\n");
        return 2;
    }

    if (!make_dirs(c.log_dir)) {
        fprintf(stderr, "error: cannot create log dir %s\n", c.log_dir.c_str());
        return 2;
    }

    const bool accept = c.mode == "accept";
    child ch;
    const auto fail = [&](const char * gate, const std::string & reason) -> int {
        if (ch.started) {
            const double waited = child_kill(ch);
            printf("cleanup: killed own child pid=%d, waited %.1fs\n", (int) ch.pid, waited);
        }
        printf("Q4E_NGRAM_FAIL %s: %s\n", gate, reason.c_str());
        ws_fini();
        return 1;
    };

    // G0: no residual process of this exe (Windows), port free
#ifdef _WIN32
    const std::string exe_name = base_name(c.server);
    if (exe_running(exe_name)) {
        return fail("G0", "residual process still running: " + exe_name);
    }
#endif
    if (port_in_use(c.host, c.port)) {
        return fail("G0", "port " + std::to_string(c.port) + " already in use");
    }
    printf("G0 PASS: no residual %s; port %d free\n", base_name(c.server).c_str(), c.port);

    // START: one server instance for the whole run
    {
        const std::vector<std::string> sargs = {
            "--model", c.model, "--host", c.host, "--port", std::to_string(c.port),
            "--api-key", c.api_key,
            "--n-gpu-layers", c.gpu_layers,
            "--q4e-moe-vram", std::to_string(c.moe_vram),
            "--ctx-size", std::to_string(c.ctx_size),
            "--parallel", "1"};
        std::string err;
        if (!child_start(c.server, sargs, c.log_dir + "/server.out.log", c.log_dir + "/server.err.log", ch, err)) {
            return fail("START", err);
        }
    }
    printf("START pid=%d\n", (int) ch.pid);

    // HEALTH: /health 200 within deadline; fail if the child exits first
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(c.health_timeout);
        bool ok = false;
        while (std::chrono::steady_clock::now() < deadline) {
            httplib::Client cli(c.host, c.port);
            cli.set_connection_timeout(5);
            cli.set_read_timeout(5);
            const auto res = cli.Get("/health");
            if (res && res->status == 200) {
                ok = true;
                break;
            }
            if (!child_alive(ch)) {
                printf("--- tail(30) %s/server.err.log ---\n", c.log_dir.c_str());
                print_tail(c.log_dir + "/server.err.log", 30);
                return fail("HEALTH", "server exited before /health 200");
            }
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
        if (!ok) {
            printf("--- tail(30) %s/server.err.log ---\n", c.log_dir.c_str());
            print_tail(c.log_dir + "/server.err.log", 30);
            return fail("HEALTH", "health timeout");
        }
        printf("HEALTH PASS: /health 200\n");
    }

    // WARMUP: fixed prompt, temp=0, non-stream
    {
        const std::string err = do_warmup(c);
        if (!err.empty()) {
            return fail("WARMUP", err);
        }
    }

    // PHASES: strictly serial, one mode switch + one timed stream each
    const char * names[4];
    const char * modes[4];
    if (accept) {
        names[0] = "P1"; modes[0] = "off";
        names[1] = "P2"; modes[1] = "full";
        names[2] = "P3"; modes[2] = "off";
        names[3] = "P4"; modes[3] = "full";
    } else {
        names[0] = "A"; modes[0] = "off";
        names[1] = "B"; modes[1] = "shim";
        names[2] = "C"; modes[2] = "full";
        names[3] = "D"; modes[3] = "full";
    }
    std::vector<phase_result> results;
    for (int i = 0; i < 4; ++i) {
        phase_result pr;
        const std::string err = run_phase(c, names[i], modes[i], accept, pr);
        if (!err.empty()) {
            return fail(names[i], err);
        }
        results.push_back(std::move(pr));
    }

    // FINAL STATS: ngram_* counters (required for the report), server still up
    nlohmann::json final_stats;
    {
        const std::string err = get_stats(c, "final", final_stats);
        if (!err.empty()) {
            return fail("STATS", err);
        }
    }

    // KILL: own PID only, wait up to 180s, confirm gone
    {
        const double waited = child_kill(ch);
#ifdef _WIN32
        if (pid_present(ch.pid)) {
            return fail("KILL", "own pid " + std::to_string((int) ch.pid) + " still present after kill");
        }
#endif
        printf("KILL PASS: own child pid=%d gone after %.1fs\n", (int) ch.pid, waited);
    }

    // report
    nlohmann::json rep;
    rep["model"] = c.model;
    rep["server"] = c.server;
    rep["host"] = c.host;
    rep["port"] = c.port;
    rep["prompt"] = c.prompt;
    rep["temp"] = 0;
    rep["warmup_tokens"] = c.warmup_tokens;
    rep["tokens"] = c.tokens;
    rep["tail"] = c.tail;
    nlohmann::json phases = nlohmann::json::array();
    for (const phase_result & p : results) {
        nlohmann::json pj;
        pj["name"] = p.name;
        pj["mode"] = p.mode;
        pj["n_tokens"] = p.n_tokens;
        pj["cum_tps"] = p.cum_tps;
        if (accept) {
            pj["tail_tps"] = p.tail_tps;
            if (p.rows_warmed >= 0.0) {
                pj["rows_warmed_delta"] = p.rows_warmed;
                pj["rows_per_token"] = p.n_tokens > 0 ? p.rows_warmed / (double) p.n_tokens : 0.0;
            }
        } else {
            pj["rolling50"] = p.rolling50;
        }
        phases.push_back(std::move(pj));
    }
    rep["phases"] = phases;

    std::string verdict = "n/a";
    if (accept) {
        const double drop_cum1 = results[0].cum_tps > 0.0
                                 ? (results[0].cum_tps - results[1].cum_tps) / results[0].cum_tps * 100.0 : 0.0;
        const double drop_cum2 = results[2].cum_tps > 0.0
                                 ? (results[2].cum_tps - results[3].cum_tps) / results[2].cum_tps * 100.0 : 0.0;
        const double drop_tail1 = results[0].tail_tps > 0.0
                                  ? (results[0].tail_tps - results[1].tail_tps) / results[0].tail_tps * 100.0 : 0.0;
        const double drop_tail2 = results[2].tail_tps > 0.0
                                  ? (results[2].tail_tps - results[3].tail_tps) / results[2].tail_tps * 100.0 : 0.0;
        nlohmann::json drops;
        drops["drop_cum1"] = drop_cum1;
        drops["drop_cum2"] = drop_cum2;
        drops["drop_tail1"] = drop_tail1;
        drops["drop_tail2"] = drop_tail2;
        rep["drops"] = drops;
        const double worst = std::max(std::max(drop_cum1, drop_cum2), std::max(drop_tail1, drop_tail2));
        verdict = worst < 1.0 ? "PASS" : "FAIL";
        printf("\n== q4e-ngram accept (per phase %d tokens, tail %d) ==\n", c.tokens, c.tail);
        printf("P1 off   cum=%.2f t/s tail=%.2f t/s\n", results[0].cum_tps, results[0].tail_tps);
        printf("P2 full  cum=%.2f t/s tail=%.2f t/s\n", results[1].cum_tps, results[1].tail_tps);
        printf("P3 off   cum=%.2f t/s tail=%.2f t/s\n", results[2].cum_tps, results[2].tail_tps);
        printf("P4 full  cum=%.2f t/s tail=%.2f t/s\n", results[3].cum_tps, results[3].tail_tps);
        printf("pair 1 (P1 vs P2): cum drop=%+.2f%%  tail drop=%+.2f%%\n", drop_cum1, drop_tail1);
        printf("pair 2 (P3 vs P4): cum drop=%+.2f%%  tail drop=%+.2f%%\n", drop_cum2, drop_tail2);
        const auto vol = [](const phase_result & p) -> std::string {
            if (p.rows_warmed < 0.0 || p.n_tokens <= 0) {
                return p.name + " n/a";
            }
            return p.name + " " + std::to_string((int) p.rows_warmed) + " rows / " + std::to_string(p.n_tokens)
                   + " tokens (= " + std::to_string((int) (p.rows_warmed / (double) p.n_tokens)) + " rows/token)";
        };
        printf("volume : %s | %s\n", vol(results[1]).c_str(), vol(results[3]).c_str());
        printf("verdict  : %s (gate: max drop < 1.0%%)\n", verdict.c_str());
    } else {
        const double a = results[0].cum_tps;
        nlohmann::json vs_a;
        printf("\n== q4e-ngram diag (per phase %d tokens) ==\n", c.tokens);
        for (int i = 0; i < 4; ++i) {
            const double pct = a > 0.0 ? (results[i].cum_tps - a) / a * 100.0 : 0.0;
            vs_a[names[i]] = pct;
            printf("%s %s = %.2f t/s  (vs A: %+.1f%%)\n", names[i], modes[i], results[i].cum_tps, pct);
        }
        rep["vs_a_pct"] = vs_a;
        printf("verdict  : n/a (attribution run, no gate)\n");
    }
    rep["verdict"] = verdict;

    {
        nlohmann::json st;
        const char * keys[] = {"ngram_prefetch", "ngram_hints", "ngram_rows_warmed",
                               "ngram_rows_dropped", "ngram_bytes_warmed"};
        for (const char * k : keys) {
            st[k] = final_stats.contains(k) ? final_stats[k] : nlohmann::json(nullptr);
        }
        rep["stats"] = st;
        printf("stats    : ngram_prefetch=%s ngram_hints=%s ngram_rows_warmed=%s ngram_rows_dropped=%s ngram_bytes_warmed=%s\n",
               rep["stats"]["ngram_prefetch"].dump().c_str(), rep["stats"]["ngram_hints"].dump().c_str(),
               rep["stats"]["ngram_rows_warmed"].dump().c_str(), rep["stats"]["ngram_rows_dropped"].dump().c_str(),
               rep["stats"]["ngram_bytes_warmed"].dump().c_str());
    }

    const std::string report_path = c.log_dir + "/" + c.mode + "_report.json";
    if (!save_file(report_path, rep.dump(2) + "\n")) {
        printf("Q4E_NGRAM_FAIL report: failed to save %s\n", report_path.c_str());
        ws_fini();
        return 1;
    }
    printf("report   : %s\n", report_path.c_str());
    printf("Q4E_NGRAM_DONE verdict=%s\n", verdict.c_str());
    ws_fini();
    return 0;
}
