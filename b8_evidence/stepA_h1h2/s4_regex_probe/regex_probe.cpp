#include <cstdio>
#include <regex>
#include <string>

static int try_case(const char * name, const char * pattern, const char * compiler) {
    try {
        std::regex re(pattern);
        bool found = std::regex_search(std::string(name), re);
        std::printf("[%s] name=%s pattern=%s => %d\n", compiler, name, pattern, (int)found);
        return found;
    } catch (const std::exception & e) {
        std::printf("[%s] name=%s pattern=%s => EXCEPTION: %s\n", compiler, name, pattern, e.what());
        return -1;
    } catch (...) {
        std::printf("[%s] name=%s pattern=%s => UNKNOWN EXCEPTION\n", compiler, name, pattern);
        return -1;
    }
}

int main() {
    const char * c = "MSVC-or-MinGW (see build line)";
    const char * name1 = "blk.0.ffn_down_exps.weight";
    const char * name2 = "blk.37.ffn_gate_up_exps.weight";
    const char * name3 = "blk.0.ffn_gate_inp.weight";
    // A: the exact q4e pattern as built by string_format("blk\\.%d%s", i, "\\.ffn_(up|down|gate|gate_up)_(ch|)exps")
    const char * patA = "blk\\.0\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
    // B: simple pattern, no empty alternative
    const char * patB = "blk\\.0\\.ffn_down_exps";
    // C: same as A but without the (ch|) group
    const char * patC = "blk\\.0\\.ffn_(up|down|gate|gate_up)_exps";
    // D: control - plain literal
    const char * patD = "blk\\.0\\.ffn";

    int rA1 = try_case(name1, patA, c);
    int rA2 = try_case(name2, patA, c);
    int rA3 = try_case(name3, patA, c); // expected: no match (gate_inp, not gate_exps)
    int rB  = try_case(name1, patB, c);
    int rC  = try_case(name1, patC, c);
    int rD  = try_case(name1, patD, c);
    std::printf("SUMMARY A1=%d A2=%d A3=%d B=%d C=%d D=%d (expect A1=1 A2=1 A3=0 B=1 C=1 D=1)\n", rA1, rA2, rA3, rB, rC, rD);
    return 0;
}
