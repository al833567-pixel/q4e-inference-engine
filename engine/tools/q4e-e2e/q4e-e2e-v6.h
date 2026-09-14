// v6 CPU end-to-end gates for q4e-e2e (A-16.7b): five gates over the common
// v6 bookkeeping components with fake backends - pool assembly, coverage path
// (oracle), pipeline stress, KV sliding window, stats/JSON. Pure CPU, no server.
#pragma once

#include <string>
#include <vector>

struct q4e_e2e_v6_gate {
    int id = 0;
    const char * name = "";
    bool ok = false;
    const char * note = "";
};

struct q4e_e2e_v6_result {
    bool all_ok = false;
    std::vector<q4e_e2e_v6_gate> gates; // fixed order: gate 1..5
};

// Runs all five gates; a gate failure is recorded (ok = false) but the run
// continues to the remaining gates. kv100k = true adds a 100-step large-stride
// window jump to gate 4.
q4e_e2e_v6_result q4e_e2e_v6_run(bool kv100k);

// Test-only: same run but the H2D backend is forced to fail (returns false),
// verifying graceful degradation - gates 2/3 fail, no crash, all_ok = false.
q4e_e2e_v6_result q4e_e2e_v6_run_h2d_fail(bool kv100k);

// Human-readable gate table (id/name/ok/note) plus the all_ok line.
std::string q4e_e2e_v6_text(const q4e_e2e_v6_result & r);
