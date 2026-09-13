#include "arg.h"
#include "common.h"
#include "download.h"
#include "llama.h"
#include "speculative.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

static void test(void) {
    common_params params;

    auto assert_output_limits = [](int32_t n_batch, int32_t n_parallel, int32_t n_draft,
                                   int32_t total, int32_t per_seq) {
        const auto limits = common_speculative_get_output_limits(n_batch, n_parallel, n_draft);
        assert(limits.total == total);
        assert(limits.per_seq == per_seq);
    };

    assert_output_limits(16, 2,  3, 8, 4);
    assert_output_limits(16, 2, -1, 2, 1);
    assert_output_limits( 6, 2,  3, 6, 4);
    assert_output_limits( 2, 1,  3, 2, 2);
    assert_output_limits(
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max());

    {
        common_params_speculative spec;
        spec.synth_len = 3.4;

        auto assert_invalid = [](const common_params_speculative & value, int32_t n_max) {
            try {
                common_speculative_synth_rates_resolve(&value, n_max);
                assert(false);
            } catch (const std::invalid_argument &) {
            }
        };

        const auto rates = common_speculative_synth_rates_resolve(&spec, 4);
        assert(rates.size() == 4);
        assert(std::abs(rates[0] - 0.80581) < 1e-5);
        assert(std::abs(rates[1] - 0.64933) < 1e-5);
        assert(std::abs(rates[2] - 0.52323) < 1e-5);
        assert(std::abs(rates[3] - 0.42163) < 1e-5);
        assert(std::abs(1.0 + rates[0] + rates[1] + rates[2] + rates[3] - 3.4) < 1e-8);

        spec.synth_len = 1.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({0.0, 0.0, 0.0, 0.0}));

        spec.synth_len = 5.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({1.0, 1.0, 1.0, 1.0}));

        spec.synth_len = 5.1;
        assert_invalid(spec, 4);

        spec.synth_len = std::numeric_limits<double>::quiet_NaN();
        assert_invalid(spec, 4);

        spec.synth_len = 0.0;
        assert_invalid(spec, 4);

        spec.synth_len = -1.0;
        spec.synth_rates = {0.8, 0.6, 0.4};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        assert(common_speculative_synth_rates_resolve(&spec, 4) == spec.synth_rates);

        spec.synth_rates = {0.8, 0.9, 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, std::numeric_limits<double>::quiet_NaN(), 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, -0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        spec.synth_len = 3.0;
        assert_invalid(spec, 4);
    }

    {
        common_params base;
        base.n_parallel = 4;
        base.n_outputs_max_per_seq = 8;

        const auto draft = common_base_params_to_speculative(base);
        assert(draft.n_outputs_max == 4);
        assert(draft.n_outputs_max_per_seq == 1);
    }

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    {
        common_params penalty_params;
        assert(penalty_params.sampling.penalty_last_n == 64);
        assert(penalty_params.sampling.dry_penalty_last_n == 64);

        argv = {"binary_name", "--repeat-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--dry-penalty-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "0"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "nan"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        const char * penalty_options[] = {"--frequency-penalty", "--presence-penalty"};
        const char * nonfinite_values[] = {"nan", "inf", "-inf"};
        for (const char * option : penalty_options) {
            for (const char * value : nonfinite_values) {
                argv = {"binary_name", option, value};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));
            }
        }
    }

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    argv = {"binary_name", "-lm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_len == 3.4);
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-rates", "0.8,0.6,0.2"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_rates == std::vector<double>({0.8, 0.6, 0.2}));
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4x"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
    }

    argv = {"binary_name", "-lm", "none"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);

    argv = {"binary_name", "-lm", "mmap"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    argv = {"binary_name", "-lm", "mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    argv = {"binary_name", "-lm", "mmap+mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    argv = {"binary_name", "-lm", "dio"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    // multi-value args (CSV)
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

    printf("test-arg-parser: test --q4e-moe-vram v5 semantics (N = expert count, valid + invalid usages)\n\n");

    {
        common_params q4e_params;
        q4e_params.model.path = "model_file.gguf"; // satisfy the required-model check (same as earlier blocks)
        assert(q4e_params.q4e_moe_vram == 5000); // v5 default: routed experts in the L1 VRAM pool

        argv = {"binary_name", "--q4e-moe-vram", "5"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
        assert(q4e_params.q4e_moe_vram == 5);

        argv = {"binary_name", "--q4e-moe-vram", "0"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
        assert(q4e_params.q4e_moe_vram == 0);

        argv = {"binary_name", "--q4e-moe-vram", "99999"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
        assert(q4e_params.q4e_moe_vram == 99999);

        argv = {"binary_name", "--q4e-moe-vram", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--q4e-moe-vram", "100000"}; // v5 range is 0-99999
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));

        // retired MoE CPU-placement flags must no longer be recognized
        argv = {"binary_name", "-ncmoe", "5"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "-cmoe"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "-ncffn", "3"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
    }

    printf("test-arg-parser: test --q4e-moe-ram (valid + invalid usages)\n\n");

    {
        common_params q4e_params;
        q4e_params.model.path = "model_file.gguf";
        assert(q4e_params.q4e_moe_ram == 4000); // v5 default: routed experts in the L2 pinned RAM pool

        argv = {"binary_name", "--q4e-moe-ram", "0"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
        assert(q4e_params.q4e_moe_ram == 0);

        argv = {"binary_name", "--q4e-moe-ram", "83"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
        assert(q4e_params.q4e_moe_ram == 83);

        argv = {"binary_name", "--q4e-moe-ram", "99999"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
        assert(q4e_params.q4e_moe_ram == 99999);

        argv = {"binary_name", "--q4e-moe-ram", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--q4e-moe-ram", "100000"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
    }

    printf("test-arg-parser: test --q4e-prefetch (valid + invalid usages)\n\n");

    {
        common_params q4e_params;
        q4e_params.model.path = "model_file.gguf"; // satisfy the required-model check (same as earlier blocks)
        assert(q4e_params.q4e_prefetch == true); // default

        argv = {"binary_name", "--q4e-prefetch", "off"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
        assert(q4e_params.q4e_prefetch == false);

        argv = {"binary_name", "--q4e-prefetch", "on"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
        assert(q4e_params.q4e_prefetch == true);

        argv = {"binary_name", "--no-q4e-prefetch"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
        assert(q4e_params.q4e_prefetch == false);

        argv = {"binary_name", "--q4e-prefetch", "on", "--no-q4e-prefetch"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
        assert(q4e_params.q4e_prefetch == false);

        argv = {"binary_name", "--q4e-prefetch", "yes"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
    }

    printf("test-arg-parser: test --q4e-ngram-prefetch (valid + invalid usages)\n\n");

    {
        common_params q4e_params;
        q4e_params.model.path = "model_file.gguf"; // satisfy the required-model check (same as earlier blocks)
        assert(q4e_params.q4e_ngram_prefetch == true); // default

        argv = {"binary_name", "--q4e-ngram-prefetch", "off"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
        assert(q4e_params.q4e_ngram_prefetch == false);

        argv = {"binary_name", "--q4e-ngram-prefetch", "on"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
        assert(q4e_params.q4e_ngram_prefetch == true);

        argv = {"binary_name", "--no-q4e-ngram-prefetch"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
        assert(q4e_params.q4e_ngram_prefetch == false);

        argv = {"binary_name", "--q4e-ngram-prefetch", "on", "--no-q4e-ngram-prefetch"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
        assert(q4e_params.q4e_ngram_prefetch == false);

        argv = {"binary_name", "--q4e-ngram-prefetch", "yes"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON));
    }

    printf("test-arg-parser: test v4.1 --q4e-moe-vram override generation (K = layer count, build 42 path, regression)\n\n");

    {
        std::vector<llama_model_tensor_buft_override> ovs;
        llm_add_q4e_moe_vram_overrides(48, 10, ovs);
        assert(ovs.size() == 38);
        ovs.clear();
        llm_add_q4e_moe_vram_overrides(48, 0, ovs);
        assert(ovs.size() == 48);
        ovs.clear();
        llm_add_q4e_moe_vram_overrides(48, 48, ovs);
        assert(ovs.size() == 0);
        ovs.clear();
        llm_add_q4e_moe_vram_overrides(48, 999, ovs);
        assert(ovs.size() == 0);
        ovs.clear();
        llm_add_q4e_moe_vram_overrides(38, 10, ovs); // k > n_layers: keep all experts in VRAM
        assert(ovs.size() == 28);
    }

    printf("test-arg-parser: test --q4e-moe-vram override insertion into padded buffer\n\n");

    {
        // common_params_parse pads tensor_buft_overrides with nulls (for llama_params_fit);
        // the q4e patterns must land before the first null terminator, or the loader
        // (null-terminated array) applies zero overrides. assert() may be compiled out
        // by header include order, so use hard checks:
        auto check = [](bool ok, const char * what) {
            if (!ok) {
                printf("test-arg-parser: CHECK FAILED: %s\n", what);
                std::exit(1);
            }
        };

        common_params q4e_params;
        q4e_params.model.path = "model_file.gguf";
        argv = {"binary_name", "--q4e-moe-vram", "10"};
        check(common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params, LLAMA_EXAMPLE_COMMON),
              "parse --q4e-moe-vram 10");
        auto & ovs = q4e_params.tensor_buft_overrides;
        check(ovs.size() == llama_max_tensor_buft_overrides(), "parse pads buffer to llama_max_tensor_buft_overrides()");
        check(ovs.front().pattern == nullptr, "padded buffer starts with null terminator");

        llm_insert_q4e_moe_vram_overrides(48, 10, ovs);
        for (int i = 0; i < 38; ++i) {
            check(ovs[i].pattern != nullptr && ovs[i].buft != nullptr, "pattern slot filled");
        }
        check(ovs[38].pattern == nullptr, "null terminator right after the 38 patterns");
        check(std::string(ovs[0].pattern)  == "blk\\.0\\.ffn_(up|down|gate|gate_up)_(ch|)exps", "pattern blk.0");
        check(std::string(ovs[37].pattern) == "blk\\.37\\.ffn_(up|down|gate|gate_up)_(ch|)exps", "pattern blk.37");

        // k >= n_layers: no patterns, buffer left as-is
        common_params q4e_params2;
        q4e_params2.model.path = "model_file.gguf";
        argv = {"binary_name", "--q4e-moe-vram", "999"};
        check(common_params_parse(argv.size(), list_str_to_char(argv).data(), q4e_params2, LLAMA_EXAMPLE_COMMON),
              "parse --q4e-moe-vram 999");
        llm_insert_q4e_moe_vram_overrides(48, 999, q4e_params2.tensor_buft_overrides);
        check(q4e_params2.tensor_buft_overrides.front().pattern == nullptr, "k >= n_layers leaves buffer null-first");
    }

    printf("test-arg-parser: test v5 --q4e-moe-vram N=expert count + fit formulas (hard checks)\n\n");

    {
        // assert() may be compiled out, so use hard checks (same pattern as the padded-buffer block)
        auto check = [](bool ok, const char * what) {
            if (!ok) {
                printf("test-arg-parser: CHECK FAILED: %s\n", what);
                std::exit(1);
            }
        };

        // per-expert bytes from the qwen4exp model: (450 + 450 + 600) MiB / 512 experts
        const int64_t MiB = 1024 * 1024;
        const int64_t expert_bytes = 1550 * MiB / 512;

        // R = ceil(N / n_layers); ceil(5000/48) = 105; pool = 105 * 48 = 5040
        const auto vram = common_q4e_fit_vram(48, 512, expert_bytes, 5000, 1024 * 1024 * 1024);
        check(vram.per_layer_pool == 105, "vram R = ceil(5000/48) = 105");
        check(vram.pool_total == 5040, "vram pool total = 105 * 48");
        check(vram.required_mib == (5040 * expert_bytes + MiB - 1) / MiB, "vram required MiB = ceil(pool * bytes / MiB)");
        check(vram.fits && vram.error.empty(), "vram fits when free is large");

        // not-fit: required > free, clear error text, no silent lowering of the pool
        const auto vram_fail = common_q4e_fit_vram(48, 512, expert_bytes, 5000, 1000);
        check(!vram_fail.fits, "vram not-fit when free is small");
        check(vram_fail.per_layer_pool == 105 && vram_fail.pool_total == 5040, "vram pool size unchanged on not-fit");
        check(vram_fail.error.find("--q4e-moe-vram") != std::string::npos, "vram error names the flag");
        check(vram_fail.error.find("does not fit") != std::string::npos, "vram error says does not fit");
        check(vram_fail.error.find("MiB free") != std::string::npos, "vram error reports free MiB");

        // exact boundary: required == free still fits
        const auto vram_edge = common_q4e_fit_vram(48, 512, expert_bytes, 5000, vram.required_mib);
        check(vram_edge.fits && vram_edge.error.empty(), "vram fits at exact boundary");

        // pool capped at the per-layer expert count (N = 99999 -> R capped at 512)
        const auto vram_cap = common_q4e_fit_vram(48, 512, expert_bytes, 99999, 1024 * 1024 * 1024);
        check(vram_cap.per_layer_pool == 512, "vram R capped at n_expert_per_layer");
        check(vram_cap.pool_total == 48 * 512, "vram pool total capped at 48 * 512");

        // N = 0: pool off, zero required
        const auto vram_zero = common_q4e_fit_vram(48, 512, expert_bytes, 0, 0);
        check(vram_zero.per_layer_pool == 0 && vram_zero.required_mib == 0 && vram_zero.fits, "vram N=0 -> zero pool, fits");

        // L2: ceil(4000/48) = 84; ceil(4033/48) = 85
        const auto ram = common_q4e_fit_ram(48, 512, expert_bytes, 4000, 1024 * 1024 * 1024);
        check(ram.per_layer_pool == 84, "ram R = ceil(4000/48) = 84");
        check(ram.pool_total == 4032, "ram pool total = 84 * 48");
        check(ram.fits && ram.error.empty(), "ram fits when free is large");

        const auto ram_fail = common_q4e_fit_ram(48, 512, expert_bytes, 4000, 1000);
        check(!ram_fail.fits, "ram not-fit when free is small");
        check(ram_fail.per_layer_pool == 84, "ram pool size unchanged on not-fit");
        check(ram_fail.error.find("--q4e-moe-ram") != std::string::npos, "ram error names the flag");
        check(ram_fail.error.find("does not fit") != std::string::npos, "ram error says does not fit");

        const auto ram85 = common_q4e_fit_ram(48, 512, expert_bytes, 4033, 1024 * 1024 * 1024);
        check(ram85.per_layer_pool == 85, "ram R = ceil(4033/48) = 85");
    }

    printf("test-arg-parser: test v6 q4e params (default / valid / invalid usages)\n\n");

    {
        // assert() may be compiled out, so use hard checks (same pattern as the v5 block)
        auto check = [](bool ok, const char * what) {
            if (!ok) {
                printf("test-arg-parser: CHECK FAILED: %s\n", what);
                std::exit(1);
            }
        };

        common_params v6p;
        v6p.model.path = "model_file.gguf"; // satisfy the required-model check (same as earlier blocks)
        check(v6p.q4e_moe_vram == 5000, "v6 default q4e_moe_vram == 5000");
        check(v6p.q4e_moe_ram == 4000, "v6 default q4e_moe_ram == 4000 (v5 unchanged)");
        check(v6p.q4e_staging_mb == 128, "v6 default q4e_staging_mb == 128");
        check(v6p.q4e_layer_floor == 10, "v6 default q4e_layer_floor == 10");
        check(v6p.q4e_ple_mmap == true, "v6 default q4e_ple_mmap == on");
        check(v6p.q4e_kv_ram == 6000, "v6 default q4e_kv_ram == 6000");

        // --q4e-moe-vram: v6 global pool semantics, parser contract unchanged from v5
        argv = {"binary_name", "--q4e-moe-vram", "24576"};
        check(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "parse --q4e-moe-vram 24576");
        check(v6p.q4e_moe_vram == 24576, "q4e_moe_vram set to 24576 (48 * 512, all experts)");
        argv = {"binary_name", "--q4e-moe-vram", "0"};
        check(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "parse --q4e-moe-vram 0");
        check(v6p.q4e_moe_vram == 0, "q4e_moe_vram 0 = pool off");
        argv = {"binary_name", "--q4e-moe-vram", "-1"};
        check(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "reject --q4e-moe-vram -1");
        argv = {"binary_name", "--q4e-moe-vram", "100000"};
        check(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "reject --q4e-moe-vram 100000");
        argv = {"binary_name", "--q4e-moe-vram", "abc"};
        check(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "reject --q4e-moe-vram abc");

        // --q4e-staging-mb
        argv = {"binary_name", "--q4e-staging-mb", "256"};
        check(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "parse --q4e-staging-mb 256");
        check(v6p.q4e_staging_mb == 256, "q4e_staging_mb set to 256");
        argv = {"binary_name", "--q4e-staging-mb", "0"};
        check(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "parse --q4e-staging-mb 0");
        check(v6p.q4e_staging_mb == 0, "q4e_staging_mb 0 = staging off");
        argv = {"binary_name", "--q4e-staging-mb", "-1"};
        check(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "reject --q4e-staging-mb -1");
        argv = {"binary_name", "--q4e-staging-mb", "100000"};
        check(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "reject --q4e-staging-mb 100000");
        argv = {"binary_name", "--q4e-staging-mb", "abc"};
        check(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "reject --q4e-staging-mb abc");

        // --q4e-layer-floor
        argv = {"binary_name", "--q4e-layer-floor", "16"};
        check(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "parse --q4e-layer-floor 16");
        check(v6p.q4e_layer_floor == 16, "q4e_layer_floor set to 16");
        argv = {"binary_name", "--q4e-layer-floor", "0"};
        check(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "parse --q4e-layer-floor 0");
        check(v6p.q4e_layer_floor == 0, "q4e_layer_floor 0 = pure global LRU");
        argv = {"binary_name", "--q4e-layer-floor", "-1"};
        check(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "reject --q4e-layer-floor -1");
        argv = {"binary_name", "--q4e-layer-floor", "100000"};
        check(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "reject --q4e-layer-floor 100000");
        argv = {"binary_name", "--q4e-layer-floor", "abc"};
        check(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "reject --q4e-layer-floor abc");

        // --q4e-ple-mmap
        argv = {"binary_name", "--q4e-ple-mmap", "off"};
        check(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "parse --q4e-ple-mmap off");
        check(v6p.q4e_ple_mmap == false, "q4e_ple_mmap off");
        argv = {"binary_name", "--q4e-ple-mmap", "on"};
        check(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "parse --q4e-ple-mmap on");
        check(v6p.q4e_ple_mmap == true, "q4e_ple_mmap on");
        argv = {"binary_name", "--q4e-ple-mmap", "yes"};
        check(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "reject --q4e-ple-mmap yes");

        // --q4e-kv-ram
        argv = {"binary_name", "--q4e-kv-ram", "8192"};
        check(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "parse --q4e-kv-ram 8192");
        check(v6p.q4e_kv_ram == 8192, "q4e_kv_ram set to 8192");
        argv = {"binary_name", "--q4e-kv-ram", "0"};
        check(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "parse --q4e-kv-ram 0");
        check(v6p.q4e_kv_ram == 0, "q4e_kv_ram 0 = off");
        argv = {"binary_name", "--q4e-kv-ram", "-1"};
        check(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "reject --q4e-kv-ram -1");
        argv = {"binary_name", "--q4e-kv-ram", "100000"};
        check(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "reject --q4e-kv-ram 100000");
        argv = {"binary_name", "--q4e-kv-ram", "abc"};
        check(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "reject --q4e-kv-ram abc");

        // combined: v6 production baseline from v6_evidence/a160_design/design.md
        argv = {"binary_name", "--q4e-moe-vram", "5000", "--q4e-moe-ram", "4000",
                "--q4e-staging-mb", "128", "--q4e-layer-floor", "10",
                "--q4e-ple-mmap", "on", "--q4e-kv-ram", "6000"};
        check(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), v6p, LLAMA_EXAMPLE_COMMON),
              "parse full v6 baseline");
        check(v6p.q4e_moe_vram == 5000 && v6p.q4e_moe_ram == 4000 && v6p.q4e_staging_mb == 128
              && v6p.q4e_layer_floor == 10 && v6p.q4e_ple_mmap == true && v6p.q4e_kv_ram == 6000,
              "v6 baseline values all set");
    }

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    setenv("LLAMA_ARG_LOAD_MODE", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_LOAD_MODE", "mmap", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    setenv("LLAMA_ARG_LOAD_MODE", "mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "mmap+mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "dio", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_LOAD_MODE", "none", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    printf("test-arg-parser: test download functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}

int main(void) {
    try {
        test();
    } catch (std::exception & e) {
        fprintf(stderr, "test-arg-parser: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
