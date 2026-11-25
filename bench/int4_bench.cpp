// Wooyoung Kim: C++ benchmark driver for the Hadamard-rotated INT4 linear operator.
//
// Runs the controlled protocol of proposal Section 6 on one thread and writes one CSV
// row per (case, mode, kernel): weight-preparation time, input-call latency, kernel-only
// time, validation and harness cost (median, Q1, Q3, IQR over the timed runs), error
// against the FP64 product of the original FP32 inputs, exact scalar/blocked agreement,
// and logical storage accounting. A sidecar <csv>.meta.txt records seeds, outlier
// columns, compiler and build settings. Optional binary dumps feed
// scripts/reference_check.py, which recomputes every error metric with NumPy FP64.
//
// See docs/benchmark.md for the column definitions and the measurement protocol.

#include "bench/accounting.hpp"
#include "bench/dump.hpp"
#include "bench/linear.hpp"
#include "bench/metrics.hpp"
#include "bench/stats.hpp"
#include "bench/workload.hpp"
#include "int4/blocked_kernel.hpp"
#include "int4/core.hpp"
#include "int4/linear.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef INT4_BENCH_BUILD_TYPE
#define INT4_BENCH_BUILD_TYPE "unspecified"
#endif
#ifndef INT4_BENCH_COMPILER
#define INT4_BENCH_COMPILER "unspecified"
#endif

namespace {

using bench::Summary;
using int4::Matrix;

constexpr const char* usage_text =
    "usage: int4_bench [options]\n"
    "  --csv PATH              output CSV (default bench.csv); metadata goes to PATH.meta.txt\n"
    "  --dump-dir DIR          write binary X/W/Y dumps for scripts/reference_check.py\n"
    "  --scales LIST           K = N values, powers of two <= 1024 (default 256,512,1024)\n"
    "  --seeds LIST            data seeds (default 0,1,2)\n"
    "  --rotation-seed N       Hadamard sign seed (default 0)\n"
    "  --warmups N             untimed warm-up calls per measurement (default 2)\n"
    "  --runs N                timed calls per measurement, >= 1 (default 5)\n"
    "  --tile T                blocked-kernel tile edge (default 32)\n"
    "  --tuning-tiles LIST     optional extension: extra blocked tile edges, run only on\n"
    "                          K = N = 512, M = 8, a = 10 (e.g. 16,64)\n"
    "  --scalar-timing MODE    protocol (M = 8, a = 10 only; default) | all | none\n"
    "  --scale-precision P     float (default) | double\n"
    "  --quick                 smoke protocol: scales 16,64; seed 0; 1 warm-up; 3 runs\n"
    "  --help                  show this message\n";

struct UsageError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

struct Options {
    std::vector<std::size_t> scales{256, 512, 1024};
    std::vector<std::uint64_t> seeds{0, 1, 2};
    std::uint64_t rotation_seed = 0;
    std::size_t warmups = 2;
    std::size_t runs = 5;
    std::size_t tile = int4::default_tile_edge;
    std::vector<std::size_t> tuning_tiles;
    std::string scalar_timing = "protocol";
    std::string scale_precision = "float";
    std::string csv_path = "bench.csv";
    std::string dump_dir;
    std::string command_line;
    bool help = false;
};

std::uint64_t parse_unsigned(const std::string& text, const std::string& flag) {
    if (text.empty() || text.size() > 19 ||
        !std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        throw UsageError(flag + " expects a non-negative integer, got '" + text + "'");
    }
    return std::stoull(text);
}

template <typename T>
std::vector<T> parse_list(const std::string& text, const std::string& flag) {
    std::vector<T> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) values.push_back(static_cast<T>(parse_unsigned(item, flag)));
    if (values.empty() || text.back() == ',') throw UsageError(flag + " expects a comma-separated list");
    return values;
}

void check_tile(std::size_t tile, const std::string& flag) {
    if (tile == 0 || tile > int4::max_tile_edge) throw UsageError(flag + " must be in [1, 1024]");
}

Options parse_args(int argc, char** argv) {
    Options options;
    std::vector<std::string> args(argv + 1, argv + argc);
    for (std::size_t i = 0; i < args.size(); ++i) {
        options.command_line += (i == 0 ? "" : " ") + args[i];
    }
    // --quick sets defaults first so explicit flags can still override them.
    if (std::find(args.begin(), args.end(), "--quick") != args.end()) {
        options.scales = {16, 64};
        options.seeds = {0};
        options.warmups = 1;
        options.runs = 3;
    }
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& flag = args[i];
        if (flag == "--quick") continue;
        if (flag == "--help" || flag == "-h") {
            options.help = true;
            continue;
        }
        if (i + 1 >= args.size()) throw UsageError("missing value for " + flag);
        const std::string& value = args[++i];
        if (flag == "--csv") options.csv_path = value;
        else if (flag == "--dump-dir") options.dump_dir = value;
        else if (flag == "--scales") options.scales = parse_list<std::size_t>(value, flag);
        else if (flag == "--seeds") options.seeds = parse_list<std::uint64_t>(value, flag);
        else if (flag == "--rotation-seed") options.rotation_seed = parse_unsigned(value, flag);
        else if (flag == "--warmups") options.warmups = static_cast<std::size_t>(parse_unsigned(value, flag));
        else if (flag == "--runs") options.runs = static_cast<std::size_t>(parse_unsigned(value, flag));
        else if (flag == "--tile") options.tile = static_cast<std::size_t>(parse_unsigned(value, flag));
        else if (flag == "--tuning-tiles") options.tuning_tiles = parse_list<std::size_t>(value, flag);
        else if (flag == "--scalar-timing") options.scalar_timing = value;
        else if (flag == "--scale-precision") options.scale_precision = value;
        else throw UsageError("unknown option " + flag);
    }
    for (std::size_t scale : options.scales) {
        if (!bench::is_power_of_two(scale) || scale > bench::max_domain_k) {
            throw UsageError("--scales values must be powers of two in [1, 1024] (K = N), got " +
                             std::to_string(scale));
        }
    }
    if (options.runs == 0 || options.runs > 1000) throw UsageError("--runs must be in [1, 1000]");
    if (options.warmups > 1000) throw UsageError("--warmups must be in [0, 1000]");
    check_tile(options.tile, "--tile");
    for (std::size_t tile : options.tuning_tiles) check_tile(tile, "--tuning-tiles");
    if (options.scalar_timing != "protocol" && options.scalar_timing != "all" &&
        options.scalar_timing != "none") {
        throw UsageError("--scalar-timing must be protocol, all, or none");
    }
    if (options.scale_precision != "float" && options.scale_precision != "double") {
        throw UsageError("--scale-precision must be float or double");
    }
    if (options.csv_path.empty()) throw UsageError("--csv must not be empty");
    return options;
}

template <typename T>
std::string join(const std::vector<T>& values) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) out << (i == 0 ? "" : ",") << values[i];
    return out.str();
}

std::string format_factor(float a) {
    std::ostringstream out;
    out << a;
    return out.str();
}

std::string utc_timestamp() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm parts{};
#if defined(_WIN32)
    gmtime_s(&parts, &now);
#else
    gmtime_r(&now, &parts);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &parts);
    return buffer;
}

// ---------------------------------------------------------------------------------
using bench::Bytes;
using bench::fp32_bytes;
using bench::int4_bytes;

struct CaseInfo {
    std::string id;
    std::uint64_t seed;
    std::uint64_t rotation_seed;
    bench::CaseSpec spec;
    std::size_t outlier_cols;
};

struct Row {
    std::string mode;
    std::string kernel;
    std::string tile; // Empty for kernels without tiles.
    Summary prep;
    Summary call;
    Summary kernel_time;
    std::string validate_median; // Empty for FP32.
    double harness_median = 0;
    bench::ErrorMetrics error;
    std::string sums_match; // "1"/"0"; empty for FP32.
    std::string y_match;
    Bytes bytes;
    double checksum = 0;
};

constexpr const char* csv_header =
    "case_id,data_seed,rotation_seed,K,N,M,a,sweep,outlier_cols,mode,kernel,tile,scale_precision,"
    "warmups,runs,prep_median_ns,prep_q1_ns,prep_q3_ns,prep_iqr_ns,call_median_ns,call_q1_ns,"
    "call_q3_ns,call_iqr_ns,kernel_median_ns,kernel_q1_ns,kernel_q3_ns,kernel_iqr_ns,"
    "validate_median_ns,harness_median_ns,max_abs_err,rel_frob_err,sums_match_scalar,"
    "y_match_scalar,weight_bytes,signs_bytes,input_bytes,act_bytes,sums_bytes,output_bytes,"
    "scratch_bytes,resident_bytes,prep_peak_bytes,call_peak_bytes,checksum";

void write_summary(std::ostream& out, const Summary& summary) {
    out << ',' << summary.median << ',' << summary.q1 << ',' << summary.q3 << ',' << summary.iqr;
}

void write_row(std::ostream& out, const Options& options, const CaseInfo& info, const Row& row) {
    out << std::fixed << std::setprecision(1);
    out << info.id << ',' << info.seed << ',' << info.rotation_seed << ',' << info.spec.k << ','
        << info.spec.n << ',' << info.spec.m << ',' << format_factor(info.spec.a) << ','
        << info.spec.sweep << ',' << info.outlier_cols << ',' << row.mode << ',' << row.kernel << ','
        << row.tile << ',' << options.scale_precision << ',' << options.warmups << ',' << options.runs;
    write_summary(out, row.prep);
    write_summary(out, row.call);
    write_summary(out, row.kernel_time);
    out << ',' << row.validate_median << ',' << row.harness_median;
    out << std::defaultfloat << std::setprecision(17);
    out << ',' << row.error.max_abs << ',' << row.error.rel_frobenius << ',' << row.sums_match << ','
        << row.y_match << ',' << row.bytes.weight << ',' << row.bytes.signs << ',' << row.bytes.input << ','
        << row.bytes.act << ',' << row.bytes.sums << ',' << row.bytes.output << ',' << row.bytes.scratch
        << ',' << row.bytes.resident() << ',' << row.bytes.prep_peak << ',' << row.bytes.call_peak << ','
        << row.checksum << '\n';
    out.flush();
}

std::string format_ns(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << value;
    return out.str();
}

template <typename Scale>
struct KernelEntry {
    std::string name;
    std::size_t tile; // 0 for the scalar kernel.
    std::unique_ptr<int4::MatmulKernel<Scale>> kernel;
    std::size_t scratch_bytes(std::size_t m) const {
        if (tile == 0) return 0;
        return static_cast<const int4::BlockedKernel<Scale>&>(*kernel).scratch_bytes(m);
    }
};

template <typename Scale>
std::vector<KernelEntry<Scale>> kernels_for_case(const Options& options, const bench::CaseSpec& spec) {
    std::vector<KernelEntry<Scale>> kernels;
    kernels.push_back({"blocked", options.tile, std::make_unique<int4::BlockedKernel<Scale>>(options.tile)});
    const bool protocol_case = spec.m == 8 && spec.a == 10.0F;
    if (options.scalar_timing == "all" || (options.scalar_timing == "protocol" && protocol_case)) {
        kernels.push_back({"scalar", 0, std::make_unique<int4::ScalarKernel<Scale>>()});
    }
    if (protocol_case && spec.k == 512 && spec.n == 512) {
        for (std::size_t tile : options.tuning_tiles) {
            if (tile == options.tile) continue;
            kernels.push_back({"blocked", tile, std::make_unique<int4::BlockedKernel<Scale>>(tile)});
        }
    }
    return kernels;
}

template <typename Scale>
int run_protocol(const Options& options) {
    namespace fs = std::filesystem;
    const fs::path csv_path(options.csv_path);
    if (csv_path.has_parent_path()) fs::create_directories(csv_path.parent_path());
    std::ofstream csv(csv_path, std::ios::trunc);
    if (!csv) throw std::runtime_error("cannot open CSV for writing: " + options.csv_path);
    std::ofstream meta(options.csv_path + ".meta.txt", std::ios::trunc);
    if (!meta) throw std::runtime_error("cannot open metadata file for writing");
    if (!options.dump_dir.empty()) fs::create_directories(options.dump_dir);
    const auto dump_path = [&](const std::string& name) { return (fs::path(options.dump_dir) / name).string(); };

#ifdef NDEBUG
    const bool release_assertions_off = true;
#else
    const bool release_assertions_off = false;
    std::cerr << "int4_bench: warning: NDEBUG is not defined; timings are not from a release build\n";
#endif
    meta << "started_utc=" << utc_timestamp() << '\n'
         << "command=int4_bench " << options.command_line << '\n'
         << "compiler=" << INT4_BENCH_COMPILER
#if defined(__VERSION__)
         << " (" << __VERSION__ << ")"
#endif
         << '\n'
         << "build_type=" << INT4_BENCH_BUILD_TYPE << '\n'
         << "ndebug=" << (release_assertions_off ? 1 : 0) << '\n'
         << "cplusplus=" << __cplusplus << '\n'
         << "clock=std::chrono::steady_clock period=" << bench::Clock::period::num << '/'
         << bench::Clock::period::den << "s\n"
         << "threads=1\n"
         << "scale_precision=" << options.scale_precision << '\n'
         << "warmups=" << options.warmups << "\nruns=" << options.runs << '\n'
         << "scales=" << join(options.scales) << "\nseeds=" << join(options.seeds) << '\n'
         << "rotation_seed=" << options.rotation_seed << "\ntile=" << options.tile << '\n'
         << "tuning_tiles=" << join(options.tuning_tiles) << '\n'
         << "scalar_timing=" << options.scalar_timing << '\n'
         << "dump_dir=" << options.dump_dir << '\n';

    csv << csv_header << '\n';
    const auto cases = bench::protocol_cases(options.scales);
    std::vector<double> samples;
    std::size_t rows = 0;
    std::size_t mismatches = 0;

    for (std::uint64_t seed : options.seeds) {
        for (std::size_t scale : options.scales) {
            // Input generation and dumping happen outside every timed region.
            const bench::ScaleData data = bench::generate_scale_data(scale, scale, seed);
            meta << "outlier_cols_K" << scale << "_s" << seed << '=' << join(data.outlier_cols) << '\n';
            if (!options.dump_dir.empty()) {
                bench::write_matrix_bin(dump_path("W_K" + std::to_string(scale) + "_s" + std::to_string(seed) + ".bin"),
                                        data.w);
            }
            for (const bench::CaseSpec& spec : cases) {
                if (spec.k != scale) continue;
                CaseInfo info{"K" + std::to_string(spec.k) + "_M" + std::to_string(spec.m) + "_a" +
                                  format_factor(spec.a) + "_s" + std::to_string(seed),
                              seed, options.rotation_seed, spec, data.outlier_cols.size()};
                std::cerr << "int4_bench: " << info.id << '\n';
                const Matrix<float> x = bench::make_case_input(data, spec.m, spec.a);
                const Matrix<double> y64 = bench::reference_fp64(x, data.w);
                if (!options.dump_dir.empty()) bench::write_matrix_bin(dump_path("X_" + info.id + ".bin"), x);

                bench::measure([] { return 0; }, options.warmups, options.runs, samples);
                const double harness = bench::summarize(samples).median;

                // ---- Scalar FP32 baseline -------------------------------------------
                {
                    Row row;
                    row.mode = "fp32";
                    row.kernel = "fp32";
                    row.harness_median = harness;
                    const bench::Fp32Linear op = bench::measure(
                        [&] {
                            bench::Fp32Linear fresh;
                            fresh.prepare(data.w);
                            return fresh;
                        },
                        options.warmups, options.runs, samples);
                    row.prep = bench::summarize(samples);
                    const Matrix<float> y = bench::measure([&] { return op.run(x); }, options.warmups,
                                                           options.runs, samples);
                    row.call = bench::summarize(samples);
                    const Matrix<float> y_kernel = bench::measure(
                        [&] { return bench::fp32_matmul(x, op.weights()); }, options.warmups, options.runs, samples);
                    row.kernel_time = bench::summarize(samples);
                    if (!bench::identical(y, y_kernel)) throw std::logic_error("FP32 path is not deterministic");
                    row.error = bench::compute_errors(y, y64);
                    row.bytes = fp32_bytes(spec.m, spec.k, spec.n);
                    row.checksum = bench::checksum(y);
                    write_row(csv, options, info, row);
                    ++rows;
                    if (!options.dump_dir.empty()) bench::write_matrix_bin(dump_path("Y_" + info.id + "_fp32.bin"), y);
                }

                // ---- Direct and rotated W4A4 ----------------------------------------
                for (int4::LinearMode mode : {int4::LinearMode::Direct, int4::LinearMode::Rotated}) {
                    const int4::QuantizedLinear<Scale> op = bench::measure(
                        [&] {
                            int4::QuantizedLinear<Scale> fresh(mode, options.rotation_seed);
                            fresh.prepare(data.w);
                            return fresh;
                        },
                        options.warmups, options.runs, samples);
                    const Summary prep = bench::summarize(samples);

                    // Identical packed operands for every kernel; the scalar kernel's
                    // result is the exact reference for the blocked kernels.
                    const int4::PackedInt4Matrix<Scale> qx = op.quantize_input(x);
                    const int4::PackedInt4Matrix<Scale>& qw = op.packed_weights();
                    const Matrix<std::int32_t> reference_sums = int4::ScalarKernel<Scale>().run(qx, qw);
                    const Matrix<float> reference_y =
                        int4::rescale_output(reference_sums, qx.scales(), qw.scales());
                    bench::measure(
                        [&] {
                            qx.validate();
                            qw.validate();
                            return 0;
                        },
                        options.warmups, options.runs, samples);
                    const std::string validate_median = format_ns(bench::summarize(samples).median);

                    for (const KernelEntry<Scale>& entry : kernels_for_case<Scale>(options, spec)) {
                        Row row;
                        row.mode = int4::mode_name(mode);
                        row.kernel = entry.name;
                        row.tile = entry.tile == 0 ? "" : std::to_string(entry.tile);
                        row.prep = prep;
                        row.validate_median = validate_median;
                        row.harness_median = harness;
                        const Matrix<float> y = bench::measure([&] { return op.run(x, *entry.kernel); },
                                                               options.warmups, options.runs, samples);
                        row.call = bench::summarize(samples);
                        const Matrix<std::int32_t> sums = bench::measure(
                            [&] { return entry.kernel->run(qx, qw); }, options.warmups, options.runs, samples);
                        row.kernel_time = bench::summarize(samples);
                        const bool sums_ok = bench::identical(sums, reference_sums);
                        const bool y_ok = bench::identical(y, reference_y);
                        row.sums_match = sums_ok ? "1" : "0";
                        row.y_match = y_ok ? "1" : "0";
                        if (!sums_ok || !y_ok) {
                            ++mismatches;
                            std::cerr << "int4_bench: MISMATCH vs scalar kernel in " << info.id << ' ' << row.mode
                                      << ' ' << row.kernel << row.tile << '\n';
                        }
                        row.error = bench::compute_errors(y, y64);
                        row.bytes = int4_bytes(spec.m, spec.k, spec.n, mode == int4::LinearMode::Rotated,
                                               op.weight_bytes(), op.signs_bytes(), qx.storage_bytes(),
                                               entry.scratch_bytes(spec.m), op.rotation_scratch_bytes());
                        row.checksum = bench::checksum(y);
                        write_row(csv, options, info, row);
                        ++rows;
                        if (!options.dump_dir.empty() && entry.name == "blocked" && entry.tile == options.tile) {
                            bench::write_matrix_bin(dump_path("Y_" + info.id + "_" + row.mode + ".bin"), y);
                        }
                    }
                }
            }
        }
    }
    meta << "finished_utc=" << utc_timestamp() << '\n'
         << "rows=" << rows << "\nkernel_mismatches=" << mismatches << '\n';
    std::cerr << "int4_bench: wrote " << rows << " rows to " << options.csv_path << " ("
              << mismatches << " kernel mismatches)\n";
    return mismatches == 0 ? 0 : 3;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_args(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "int4_bench: " << error.what() << '\n' << usage_text;
        return 2;
    }
    if (options.help) {
        std::cout << usage_text;
        return 0;
    }
    try {
        return options.scale_precision == "double" ? run_protocol<double>(options) : run_protocol<float>(options);
    } catch (const std::exception& error) {
        std::cerr << "int4_bench: error: " << error.what() << '\n';
        return 1;
    }
}
