#pragma once

// Member 3: timing harness and summary statistics for the benchmark driver.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace bench {

using Clock = std::chrono::steady_clock;
static_assert(Clock::is_steady, "benchmark clock must be monotonic");

struct Summary {
    double median = 0;
    double q1 = 0;
    double q3 = 0;
    double iqr = 0;
    std::size_t count = 0;
};

// Linear-interpolation quantile of sorted data (NumPy's default "linear" method).
// For five runs, q1/median/q3 are exactly the 2nd/3rd/4th order statistics.
inline double quantile_sorted(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) throw std::invalid_argument("quantile of an empty sample");
    if (!(p >= 0 && p <= 1)) throw std::invalid_argument("quantile probability must be in [0,1]");
    const double position = p * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

inline Summary summarize(std::vector<double> samples) {
    if (samples.empty()) throw std::invalid_argument("cannot summarize an empty sample");
    for (double sample : samples) {
        if (!std::isfinite(sample)) throw std::invalid_argument("nonfinite timing sample");
    }
    std::sort(samples.begin(), samples.end());
    Summary summary;
    summary.median = quantile_sorted(samples, 0.5);
    summary.q1 = quantile_sorted(samples, 0.25);
    summary.q3 = quantile_sorted(samples, 0.75);
    summary.iqr = summary.q3 - summary.q1;
    summary.count = samples.size();
    return summary;
}

// Runs fn() warmups+runs times on one thread and records the last `runs` durations in
// nanoseconds. Only the call itself is inside the timed region; each result is kept
// alive until the next call returns, so destroying it is never timed. The final result
// is returned so the caller can consume it after timing (prevents dead-code removal and
// lets errors be computed from a timed output).
template <typename Fn>
auto measure(Fn&& fn, std::size_t warmups, std::size_t runs, std::vector<double>& samples_ns) {
    using Result = std::decay_t<decltype(fn())>;
    if (runs == 0) throw std::invalid_argument("at least one timed run is required");
    samples_ns.clear();
    samples_ns.reserve(runs);
    std::optional<Result> last;
    for (std::size_t i = 0; i < warmups + runs; ++i) {
        const auto start = Clock::now();
        Result result = fn();
        const auto stop = Clock::now();
        if (i >= warmups) {
            samples_ns.push_back(std::chrono::duration<double, std::nano>(stop - start).count());
        }
        last.reset();
        last.emplace(std::move(result));
    }
    return std::move(*last);
}

} // namespace bench
