#pragma once

// Qinyan Zeng: prepared FP32 / direct W4A4 / rotated W4A4 operator.
// Reuse Wooyoung Kim's independent scalar FP32 baseline.
#include "bench/linear.hpp"
#include "int4/hadamard.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace int4 {

enum class LinearMode { Fp32, Direct, Rotated };

inline const char* mode_name(LinearMode mode) {
    switch (mode) {
    case LinearMode::Fp32: return "fp32";
    case LinearMode::Direct: return "direct";
    case LinearMode::Rotated: return "rotated";
    }
    throw std::invalid_argument("unknown linear mode");
}

template <typename Scale = float>
class QuantizedLinear {
    static_assert(std::is_same_v<Scale, float> || std::is_same_v<Scale, double>,
                  "scale precision must be float or double");
public:
    explicit QuantizedLinear(LinearMode mode = LinearMode::Rotated, std::uint64_t rotation_seed = 0)
        : mode_(mode), rotation_seed_(rotation_seed) {
        (void)mode_name(mode_);
    }

    LinearMode mode() const noexcept { return mode_; }
    std::uint64_t rotation_seed() const noexcept { return rotation_seed_; }
    bool prepared() const noexcept { return static_cast<bool>(state_); }

    void prepare(const Matrix<float>& w) {
        check_weight_domain(w.rows(), w.cols());
        check_finite(w, "W");
        auto next = std::make_unique<State>();
        next->k = w.rows();
        if (mode_ == LinearMode::Fp32) {
            next->fp32.emplace();
            next->fp32->prepare(w);
        } else if (mode_ == LinearMode::Rotated) {
            next->rotation.emplace(w.rows(), rotation_seed_);
            Matrix<float> rotated = w;
            next->rotation->apply_transpose_left(rotated);
            next->packed.emplace(quantize_int4<Scale>(rotated, ScaleAxis::Column));
        } else {
            next->packed.emplace(quantize_int4<Scale>(w, ScaleAxis::Column));
        }
        // The only mutation of prepared state is noexcept. Validation, allocation,
        // rotation overflow and quantization underflow all preserve the old state.
        state_.swap(next);
    }

    PackedInt4Matrix<Scale> quantize_input(const Matrix<float>& x) const {
        const State& state = ready();
        if (mode_ == LinearMode::Fp32) throw std::logic_error("fp32 mode has no packed inputs");
        check_input_domain(x.rows(), x.cols(), state.k);
        check_finite(x, "X");
        if (state.rotation) {
            Matrix<float> rotated = x;
            state.rotation->apply_right(rotated);
            return quantize_int4<Scale>(rotated, ScaleAxis::Row);
        }
        return quantize_int4<Scale>(x, ScaleAxis::Row);
    }

    Matrix<float> run(const Matrix<float>& x, const MatmulKernel<Scale>& kernel) const {
        const State& state = ready();
        if (mode_ == LinearMode::Fp32) {
            auto output = state.fp32->run(x);
            for (std::size_t i = 0; i < output.size(); ++i) {
                (void)checked_fp32(output.data()[i]);
            }
            return output;
        }
        const auto qx = quantize_input(x);
        const auto sums = kernel.run(qx, *state.packed);
        return rescale_output(sums, qx.scales(), state.packed->scales());
    }

    const PackedInt4Matrix<Scale>& packed_weights() const {
        const State& state = ready();
        if (!state.packed) throw std::logic_error("fp32 mode has no packed weights");
        return *state.packed;
    }

    const HadamardRotation* rotation() const noexcept {
        return state_ && state_->rotation ? &*state_->rotation : nullptr;
    }

    std::size_t weight_bytes() const {
        const State& state = ready();
        return state.fp32 ? state.fp32->resident_bytes() : state.packed->storage_bytes();
    }
    std::size_t signs_bytes() const noexcept {
        const auto* value = rotation();
        return value ? value->storage_bytes() : 0;
    }
    std::size_t rotation_scratch_bytes() const noexcept {
        const auto* value = rotation();
        return value ? value->scratch_bytes() : 0;
    }

private:
    struct State {
        std::size_t k = 0;
        std::optional<HadamardRotation> rotation;
        std::optional<PackedInt4Matrix<Scale>> packed;
        std::optional<bench::Fp32Linear> fp32;
    };

    const State& ready() const {
        if (!state_) throw std::logic_error("operator used before prepare()");
        return *state_;
    }

    LinearMode mode_;
    std::uint64_t rotation_seed_;
    std::unique_ptr<State> state_; // Move-only ownership; one atomic prepared-state replacement.
};

} // namespace int4
