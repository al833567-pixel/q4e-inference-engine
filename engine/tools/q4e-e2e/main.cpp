// q4e-e2e: end-to-end acceptance client for q4e-server (replaces a9_e2e.ps1)

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
    int         port = 8123;
    std::string api_key = "a141key";
    int         ctx_size = 2048;
    int         max_tokens = 64;
    int         health_timeout = 480;
    std::string log_dir = "q4e-e2e-run";
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

static void usage() {
    printf("q4e-e2e: end-to-end acceptance client for q4e-server\n");
    printf("\n");
    printf("usage: q4e-e2e --server <q4e-server-exe> --model <gguf> [options]\n");
    printf("\n");
    printf("required:\n");
    printf("  --server <path>      path to q4e-server binary\n");
    printf("  --model <path>       path to GGUF model file\n");
    printf("\n");
    printf("options:\n");
    printf("  --host <ip>          default: 127.0.0.1\n");
    printf("  --port <n>           default: 8123\n");
    printf("  --api-key <key>      default: a141key\n");
    printf("  --ctx-size <n>       default: 2048\n");
    printf("  --max-tokens <n>     default: 64\n");
    printf("  --health-timeout <s> G1 deadline in seconds, default: 480\n");
    printf("  --log-dir <dir>      evidence dir, created if missing, default: q4e-e2e-run\n");
    printf("  --help               show this help\n");
    printf("\n");
    printf("notes:\n");
    printf("  - kills only the child PID it spawned; never touches other processes\n");
    printf("  - on non-Windows, G0 checks only the port (no process scan)\n");
    printf("  - exit 0 + Q4E_E2E_ALL_PASS on success, exit 1 + Q4E_E2E_FAIL <gate> on failure\n");
}

static std::string base_name(const std::string & p) {
    const size_t i = p.find_last_of("/\\");
    return i == std::string::npos ? p : p.substr(i + 1);
}

static std::string join(const std::vector<std::string> & v, const std::string & sep) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) {
            out += sep;
        }
        out += v[i];
    }
    return out;
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

// count SSE "data:" lines and check for a trimmed "data: [DONE]"
static void sse_check(const std::string & body, size_t & data_lines, bool & done) {
    data_lines = 0;
    done = false;
    size_t pos = 0;
    while (pos < body.size()) {
        const size_t eol = body.find('\n', pos);
        const std::string raw = body.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
        const std::string line = trim(raw);
        if (line.rfind("data:", 0) == 0) {
            ++data_lines;
            if (line == "data: [DONE]") {
                done = true;
            }
        }
        if (eol == std::string::npos) {
            break;
        }
        pos = eol + 1;
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

static bool port_in_use(const conf & c) {
#ifdef _WIN32
    httplib::Client cli(c.host, c.port);
    cli.set_connection_timeout(2);
    cli.set_read_timeout(2);
    return cli.Get("/") != nullptr;
#else
    return port_in_use(c.host, c.port);
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

static bool child_start(const conf & c, child & ch, std::string & err) {
    const std::string out_log = c.log_dir + "/server.out.log";
    const std::string err_log = c.log_dir + "/server.err.log";
#ifdef _WIN32
    const auto q = [](const std::string & s) { return std::string("\"") + s + "\""; };
    std::string cmd;
    cmd += q(c.server) + " --model " + q(c.model);
    cmd += " --host " + c.host + " --port " + std::to_string(c.port);
    cmd += " --api-key " + c.api_key + " --ctx-size " + std::to_string(c.ctx_size);
    cmd += " --parallel 1";
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
        std::vector<std::string> a = {
            c.server,
            "--model", c.model,
            "--host", c.host,
            "--port", std::to_string(c.port),
            "--api-key", c.api_key,
            "--ctx-size", std::to_string(c.ctx_size),
            "--parallel", "1",
        };
        std::vector<char *> av;
        for (std::string & s : a) {
            av.push_back(&s[0]);
        }
        av.push_back(nullptr);
        execv(c.server.c_str(), av.data());
        _exit(127);
    }
    ::close(out);
    ::close(errf);
    ch.pid = pid;
    ch.started = true;
    return true;
#endif
}

static nlohmann::json chat_body(const char * content, int max_tokens, bool stream) {
    nlohmann::json j;
    j["messages"] = nlohmann::json::array();
    nlohmann::json msg;
    msg["role"] = "user";
    msg["content"] = content;
    j["messages"].push_back(std::move(msg));
    j["max_tokens"] = max_tokens;
    j["temperature"] = 0;
    j["stream"] = stream;
    return j;
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
            } else if (a == "--max-tokens") {
                c.max_tokens = std::stoi(value("--max-tokens"));
            } else if (a == "--health-timeout") {
                c.health_timeout = std::stoi(value("--health-timeout"));
            } else if (a == "--log-dir") {
                c.log_dir = value("--log-dir");
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
    if (c.server.empty() || c.model.empty()) {
        fprintf(stderr, "error: --server and --model are required (see --help)\n");
        return 2;
    }

    if (!make_dirs(c.log_dir)) {
        fprintf(stderr, "error: cannot create log dir %s\n", c.log_dir.c_str());
        return 2;
    }

    child ch;
    const auto fail = [&](const char * gate, const std::string & reason) -> int {
        printf("%s FAIL: %s\n", gate, reason.c_str());
        if (ch.started) {
            const double waited = child_kill(ch);
            printf("cleanup: killed own child, waited %.1fs\n", waited);
        }
        printf("Q4E_E2E_FAIL %s\n", gate);
        return 1;
    };

    // G0: no residual server process (Windows only), port free
#ifdef _WIN32
    const std::string exe_name = base_name(c.server);
    if (exe_running(exe_name)) {
        return fail("G0", "residual process: " + exe_name);
    }
#else
    // non-Windows: no process scan, port check below is the G0 gate
#endif
    if (port_in_use(c)) {
        return fail("G0", "port " + std::to_string(c.port) + " already in use");
    }
    printf("G0 PASS: no residual server process; port %d free\n", c.port);

    // START: spawn q4e-server with logs redirected
    {
        std::string err;
        if (!child_start(c, ch, err)) {
            printf("START FAIL: %s\n", err.c_str());
            printf("Q4E_E2E_FAIL START\n");
            return 1;
        }
    }
    printf("START pid=%d\n", (int) ch.pid);

    // G1: /health 200 within deadline; fail if the child exits first
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
                printf("G1 FAIL: server process exited before /health 200\n");
                printf("--- tail(30) %s/server.err.log ---\n", c.log_dir.c_str());
                print_tail(c.log_dir + "/server.err.log", 30);
                printf("Q4E_E2E_FAIL G1\n");
                return 1;
            }
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
        if (!ok) {
            printf("G1 FAIL: /health never returned 200 within %ds\n", c.health_timeout);
            printf("--- tail(30) %s/server.err.log ---\n", c.log_dir.c_str());
            print_tail(c.log_dir + "/server.err.log", 30);
            return fail("G1", "health timeout");
        }
        printf("G1 PASS: /health 200\n");
    }

    // G2: POST chat completion without auth -> 401
    {
        const std::string body = chat_body("What is a cache? Explain in one sentence.", c.max_tokens, false).dump();
        httplib::Client cli(c.host, c.port);
        cli.set_connection_timeout(10);
        cli.set_read_timeout(30);
        const auto res = cli.Post("/v1/chat/completions", body, "application/json");
        if (!res) {
            return fail("G2", "no response (expected 401)");
        }
        if (res->status != 401) {
            return fail("G2", "expected 401, got " + std::to_string(res->status));
        }
        printf("G2 PASS: 401 without auth\n");
    }

    // G3: authed completion -> 200 + completion_tokens >= 1
    {
        const std::string body = chat_body("What is a cache? Explain in one sentence.", c.max_tokens, false).dump();
        httplib::Client cli(c.host, c.port);
        cli.set_connection_timeout(10);
        cli.set_read_timeout(900);
        const httplib::Headers headers{{"Authorization", "Bearer " + c.api_key}};
        const auto res = cli.Post("/v1/chat/completions", headers, body, "application/json");
        if (!res) {
            return fail("G3", "no response (expected 200)");
        }
        if (res->status != 200) {
            return fail("G3", "expected 200, got " + std::to_string(res->status));
        }
        if (!save_file(c.log_dir + "/gen.json", res->body)) {
            return fail("G3", "failed to save gen.json");
        }
        const auto parsed = nlohmann::json::parse(res->body, nullptr, false);
        long long completion_tokens = 0;
        if (!parsed.is_discarded() && parsed.contains("usage") && parsed["usage"].contains("completion_tokens")
            && parsed["usage"]["completion_tokens"].is_number()) {
            completion_tokens = parsed["usage"]["completion_tokens"].get<long long>();
        }
        if (completion_tokens < 1) {
            return fail("G3", "completion_tokens < 1 (got " + std::to_string(completion_tokens) + ")");
        }
        std::string line = "G3 PASS: 200 completion_tokens=" + std::to_string(completion_tokens);
        if (!parsed.is_discarded() && parsed.contains("time_total")) {
            line += " time_total=" + parsed["time_total"].dump();
        }
        printf("%s\n", line.c_str());
    }

    // G4: SSE stream -> >= 2 data: lines + one data: [DONE]
    {
        const std::string body = chat_body("Say hello in one short sentence.", 8, true).dump();
        httplib::Client cli(c.host, c.port);
        cli.set_connection_timeout(10);
        cli.set_read_timeout(300);
        const httplib::Headers headers{{"Authorization", "Bearer " + c.api_key}};
        const auto res = cli.Post("/v1/chat/completions", headers, body, "application/json");
        if (!res) {
            return fail("G4", "no response");
        }
        if (!save_file(c.log_dir + "/sse.txt", res->body)) {
            return fail("G4", "failed to save sse.txt");
        }
        size_t data_lines = 0;
        bool done = false;
        sse_check(res->body, data_lines, done);
        if (data_lines < 2 || !done) {
            const std::string snippet = res->body.substr(0, 200);
            return fail("G4", std::to_string(data_lines) + " data: lines, done=" + (done ? "true" : "false")
                        + "; raw=" + snippet);
        }
        printf("G4 PASS: SSE %zu data lines + [DONE]\n", data_lines);
    }

    // G5: /q4e/stats with auth -> prefetch gates
    {
        httplib::Client cli(c.host, c.port);
        cli.set_connection_timeout(10);
        cli.set_read_timeout(30);
        const httplib::Headers headers{{"Authorization", "Bearer " + c.api_key}};
        const auto res = cli.Get("/q4e/stats", headers);
        if (!res) {
            return fail("G5", "GET /q4e/stats: no response");
        }
        if (res->status != 200) {
            return fail("G5", "GET /q4e/stats: expected 200, got " + std::to_string(res->status));
        }
        if (!save_file(c.log_dir + "/stats.json", res->body)) {
            return fail("G5", "failed to save stats.json");
        }
        const auto j = nlohmann::json::parse(res->body, nullptr, false);
        if (j.is_discarded()) {
            return fail("G5", "stats.json is not valid JSON");
        }
        const auto num = [&j](const char * k) -> long long {
            if (j.contains(k) && j[k].is_number()) {
                return j[k].get<long long>();
            }
            return 0;
        };
        const long long q4e_prefetch = num("q4e_prefetch");
        const long long touches = num("touches");
        const long long layers_touched = num("layers_touched");
        const long long layers = num("layers");
        const long long io_issued = num("prefetch_io_issued");
        const long long overlap_hits = num("prefetch_overlap_hits");
        const long long q4e_moe_vram = num("q4e_moe_vram");
        const long long hook_skips_device = num("hook_skips_weights_device");
        std::vector<std::string> bad;
        if (q4e_prefetch != 1) {
            bad.push_back("q4e_prefetch=" + std::to_string(q4e_prefetch) + " (want 1)");
        }
        if (touches <= 0) {
            bad.push_back("touches=" + std::to_string(touches) + " (want >0)");
        }
        if (layers <= 0) {
            bad.push_back("layers=" + std::to_string(layers) + " (want >0)");
        }
        // touched books host-resident MoE layers only; VRAM-resident layers are
        // skipped (hook_skips_weights_device). Expected touched = layers minus
        // the device-resident MoE layers (K when placement active, else 0).
        const long long expected_touched =
            (hook_skips_device > 0)
                ? ((q4e_moe_vram >= layers) ? 0 : (layers - q4e_moe_vram))
                : layers;
        if (layers_touched != expected_touched) {
            bad.push_back("layers_touched=" + std::to_string(layers_touched) +
                          " (want expected=" + std::to_string(expected_touched) +
                          "; layers=" + std::to_string(layers) +
                          ", q4e_moe_vram=" + std::to_string(q4e_moe_vram) +
                          ", hook_skips_weights_device=" + std::to_string(hook_skips_device) + ")");
        }
        if (io_issued <= 0) {
            bad.push_back("prefetch_io_issued=" + std::to_string(io_issued) + " (want >0)");
        }
        if (overlap_hits <= 0) {
            bad.push_back("prefetch_overlap_hits=" + std::to_string(overlap_hits) + " (want >0)");
        }
        if (!bad.empty()) {
            return fail("G5", join(bad, " | "));
        }
        printf("G5 PASS: q4e_prefetch=1 touches=%lld layers_touched=%lld/%lld (expected %lld) prefetch_io_issued=%lld prefetch_overlap_hits=%lld\n",
               touches, layers_touched, layers, expected_touched, io_issued, overlap_hits);
        const auto opt = [&j](const char * k) -> std::string {
            if (j.contains(k)) {
                return std::string(k) + "=" + j[k].dump();
            }
            return std::string(k) + "=<absent>";
        };
        printf("G5 ungated: %s %s %s %s %s %s %s %s %s\n",
               opt("prefetch_io_bytes").c_str(), opt("prefetch_io_completed").c_str(),
               opt("prefetch_io_dropped").c_str(), opt("prefetch_events_dropped").c_str(),
               opt("prefetch_overlap_ratio").c_str(), opt("hook_skips_weights_device").c_str(),
               opt("hits_vram").c_str(), opt("hits_ram").c_str(), opt("hits_ssd").c_str());
    }

    // G6: kill only our own child PID, wait up to 180s, confirm gone
    {
        const double waited = child_kill(ch);
#ifdef _WIN32
        if (pid_present(ch.pid)) {
            printf("G6 FAIL: own pid %d still present after %.1fs wait\n", (int) ch.pid, waited);
            printf("Q4E_E2E_FAIL G6\n");
            return 1;
        }
#else
        (void) waited;
#endif
        printf("G6 PASS: own child pid=%d gone after %.1fs\n", (int) ch.pid, waited);
    }

    printf("Q4E_E2E_ALL_PASS\n");
    return 0;
}
