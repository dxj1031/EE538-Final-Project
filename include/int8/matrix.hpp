#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace int8 {

// Owns contiguous row-major storage. Copies own their data; moves empty the source.
template <typename T>
class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols)
        : data_(checked_size(rows, cols)), rows_(rows), cols_(cols) {}

    Matrix(std::size_t rows, std::size_t cols, std::vector<T> values)
        : data_(std::move(values)), rows_(rows), cols_(cols) {
        if (data_.size() != checked_size(rows, cols)) {
            throw std::invalid_argument("matrix data size does not match shape");
        }
    }

    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix(Matrix&& other) noexcept
        : data_(std::move(other.data_)),
          rows_(std::exchange(other.rows_, 0)), cols_(std::exchange(other.cols_, 0)) {
        other.data_.clear();
    }
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            rows_ = std::exchange(other.rows_, 0);
            cols_ = std::exchange(other.cols_, 0);
            other.data_.clear();
        }
        return *this;
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return data_.size(); }
    // Logical element bytes only: excludes capacity, allocator, and object overhead.
    std::size_t storage_bytes() const noexcept { return size() * sizeof(T); }
    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    T* row_data(std::size_t row) {
        check_row(row);
        return data_.data() + row * cols_;
    }
    const T* row_data(std::size_t row) const {
        check_row(row);
        return data_.data() + row * cols_;
    }
    T& operator()(std::size_t row, std::size_t col) {
        check_col(col);
        return row_data(row)[col];
    }
    const T& operator()(std::size_t row, std::size_t col) const {
        check_col(col);
        return row_data(row)[col];
    }

private:
    static std::size_t checked_size(std::size_t rows, std::size_t cols) {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("matrix dimensions must be positive");
        }
        if (rows > std::numeric_limits<std::size_t>::max() / cols) {
            throw std::length_error("matrix element count overflows size_t");
        }
        return rows * cols;
    }
    void check_row(std::size_t row) const {
        if (row >= rows_) throw std::out_of_range("matrix row out of range");
    }
    void check_col(std::size_t col) const {
        if (col >= cols_) throw std::out_of_range("matrix column out of range");
    }

    // Storage comes first so a throwing copy allocation leaves the shape unchanged.
    std::vector<T> data_;
    std::size_t rows_;
    std::size_t cols_;
};

} // namespace int8
