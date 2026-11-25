// Qinyan Zeng: file-to-file FP32 / direct / rotated W4A4 linear operator.
#include "bench/accounting.hpp"
#include "bench/stats.hpp"
#include "int4/blocked_kernel.hpp"
#include "int4/io.hpp"
#include "int4/linear.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;

constexpr const char* usage =
    "usage: int4_cli --x X.txt --w W.txt --out Y.txt --csv timings.csv [options]\n"
    "  --mode fp32|direct|rotated       default rotated\n"
    "  --kernel scalar|blocked         default blocked; unused in fp32 mode\n"
    "  --seed N                        rotation sign seed, uint64, default 0\n"
    "  --scale-precision float|double   scale storage, default float\n"
    "  --warmups N                     untimed calls, 0..1000, default 2\n"
    "  --runs N                        timed calls, 1..1000, default 5\n"
    "Files: rows columns, then exactly rows*columns finite row-major FP32 values.\n";

struct Options {
    std::string x_path;
    std::string w_path;
    std::string out_path;
    std::string csv_path;
    int4::LinearMode mode = int4::LinearMode::Rotated;
    std::string kernel = "blocked";
    std::string precision = "float";
    std::uint64_t seed = 0;
    std::size_t warmups = 2;
    std::size_t runs = 5;
};

Options parse_args(int argc, char** argv) {
    Options options;
    std::vector<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        for (const auto& previous : seen) {
            if (flag == previous) throw std::invalid_argument("duplicate option " + flag);
        }
        seen.push_back(flag);
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + flag);
        const std::string value = argv[++i];
        if (flag == "--x") options.x_path = value;
        else if (flag == "--w") options.w_path = value;
        else if (flag == "--out") options.out_path = value;
        else if (flag == "--csv") options.csv_path = value;
        else if (flag == "--mode") {
            if (value == "fp32") options.mode = int4::LinearMode::Fp32;
            else if (value == "direct") options.mode = int4::LinearMode::Direct;
            else if (value == "rotated") options.mode = int4::LinearMode::Rotated;
            else throw std::invalid_argument("--mode must be fp32, direct or rotated");
        } else if (flag == "--kernel") {
            if (value != "scalar" && value != "blocked") {
                throw std::invalid_argument("--kernel must be scalar or blocked");
            }
            options.kernel = value;
        } else if (flag == "--scale-precision") {
            if (value != "float" && value != "double") {
                throw std::invalid_argument("--scale-precision must be float or double");
            }
            options.precision = value;
        } else if (flag == "--seed") {
            options.seed = int4::parse_unsigned(value, "--seed");
        } else if (flag == "--warmups" || flag == "--runs") {
            const auto count = int4::parse_unsigned(value, flag.c_str());
            if (count > 1000 || (flag == "--runs" && count == 0)) {
                throw std::invalid_argument(flag + " is outside the supported range");
            }
            if (flag == "--runs") options.runs = static_cast<std::size_t>(count);
            else options.warmups = static_cast<std::size_t>(count);
        } else {
            throw std::invalid_argument("unknown option " + flag);
        }
    }
    if (options.x_path.empty() || options.w_path.empty() || options.out_path.empty() || options.csv_path.empty()) {
        throw std::invalid_argument("--x, --w, --out and --csv are required");
    }
    const std::vector<fs::path> inputs{options.x_path, options.w_path};
    const std::vector<fs::path> outputs{options.out_path, options.csv_path};
    const auto same = [](const fs::path& a, const fs::path& b) {
        return fs::weakly_canonical(a) == fs::weakly_canonical(b) ||
               (fs::exists(a) && fs::exists(b) && fs::equivalent(a, b));
    };
    if (same(outputs[0], outputs[1])) throw std::invalid_argument("Y and CSV paths must differ");
    for (const auto& output : outputs) {
        for (const auto& input : inputs) {
            if (same(output, input)) throw std::invalid_argument("output must not overwrite an input file");
        }
    }
    return options;
}

int4::Matrix<float> read(const std::string& path, int4::MatrixFileKind kind) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open matrix: " + path);
    try { return int4::read_matrix_text(input, kind); }
    catch (const std::exception& error) { throw std::runtime_error(path + ": " + error.what()); }
}

void make_parent(const std::string& path) {
    const fs::path parent = fs::path(path).parent_path();
    if (!parent.empty()) fs::create_directories(parent);
}

void write_summary(std::ostream& stream, const bench::Summary& summary) {
    stream << ',' << summary.median << ',' << summary.q1 << ',' << summary.q3 << ',' << summary.iqr;
}

template <typename Scale>
int execute(const Options& options) {
    const auto x = read(options.x_path, int4::MatrixFileKind::Input);
    const auto w = read(options.w_path, int4::MatrixFileKind::Weight);
    int4::check_input_domain(x.rows(), x.cols(), w.rows());
    const int4::BlockedKernel<Scale> blocked;
    std::unique_ptr<int4::MatmulKernel<Scale>> kernel;
    if (options.kernel == "scalar") kernel = std::make_unique<int4::ScalarKernel<Scale>>();
    else kernel = std::make_unique<int4::BlockedKernel<Scale>>();
    std::vector<double> samples;
    const auto op = bench::measure([&] {
        int4::QuantizedLinear<Scale> fresh(options.mode, options.seed);
        fresh.prepare(w);
        return fresh;
    }, options.warmups, options.runs, samples);
    const auto prepare_time = bench::summarize(samples);
    const auto y = bench::measure([&] { return op.run(x, *kernel); },
                                 options.warmups, options.runs, samples);
    const auto call_time = bench::summarize(samples);
    const bool fp32 = options.mode == int4::LinearMode::Fp32;
    bench::Bytes bytes;
    if (fp32) {
        bytes = bench::fp32_bytes(x.rows(), x.cols(), w.cols());
    } else {
        const auto qx = op.quantize_input(x); // Outside timed regions; storage accounting only.
        bytes = bench::int4_bytes(x.rows(), x.cols(), w.cols(), options.mode == int4::LinearMode::Rotated,
            op.weight_bytes(), op.signs_bytes(), qx.storage_bytes(),
            options.kernel == "blocked" ? blocked.scratch_bytes(x.rows()) : 0,
            op.rotation_scratch_bytes());
    }
    make_parent(options.out_path);
    make_parent(options.csv_path);
    std::ofstream output(options.out_path);
    if (!output) throw std::runtime_error("cannot open output: " + options.out_path);
    int4::write_matrix_text(output, y);
    output.close();
    if (!output) throw std::runtime_error("failed to finish matrix output");
    std::ofstream csv(options.csv_path);
    if (!csv) throw std::runtime_error("cannot open CSV: " + options.csv_path);
    csv.imbue(std::locale::classic());
    csv << "mode,kernel,scale_precision,rotation_seed,M,K,N,warmups,runs,"
           "prep_median_ns,prep_q1_ns,prep_q3_ns,prep_iqr_ns,"
           "call_median_ns,call_q1_ns,call_q3_ns,call_iqr_ns,"
           "weight_bytes,signs_bytes,input_bytes,act_bytes,sums_bytes,output_bytes,"
           "scratch_bytes,rotation_scratch_bytes,resident_bytes,prep_peak_bytes,call_peak_bytes\n";
    csv << std::setprecision(17) << int4::mode_name(options.mode) << ','
        << (fp32 ? "fp32" : options.kernel) << ',' << options.precision << ',' << options.seed << ','
        << x.rows() << ',' << x.cols() << ',' << w.cols() << ',' << options.warmups << ',' << options.runs;
    write_summary(csv, prepare_time);
    write_summary(csv, call_time);
    csv << ',' << bytes.weight << ',' << bytes.signs << ',' << bytes.input << ',' << bytes.act << ','
        << bytes.sums << ',' << bytes.output << ',' << bytes.scratch << ',' << op.rotation_scratch_bytes()
        << ',' << bytes.resident() << ',' << bytes.prep_peak << ',' << bytes.call_peak << '\n';
    csv.close();
    if (!csv) throw std::runtime_error("failed to finish CSV output");
    std::cout << "Wrote " << y.rows() << 'x' << y.cols() << " Y to " << options.out_path
              << " and timings/storage to " << options.csv_path << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << usage;
        return 0;
    }
    try {
        const auto options = parse_args(argc, argv);
        // Precision is dispatched once, outside every numerical loop.
        return options.precision == "double" ? execute<double>(options) : execute<float>(options);
    } catch (const std::exception& error) {
        std::cerr << "int4_cli: " << error.what() << '\n' << usage;
        return 1;
    }
}
