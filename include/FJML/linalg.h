// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef LINALG_INCLUDED
#define LINALG_INCLUDED

#include <cstdlib>
#include <random>

#pragma GCC target("avx2,fma")
#pragma GCC optimize("O3,unroll-loops")

#include "tensor.h"

/**
 * @brief Helper method to print the shape of a tensor.
 * @param a The tensor.
 * @return The shape of the tensor.
 */
template <typename T> static std::string print_shape(const FJML::Tensor<T>& a) {
    std::string res = "(";
    for (int i = 0; i < a.dim(); i++) {
        res += std::to_string(a.shape[i]) + ", ";
    }
    res.pop_back();
    res.pop_back();
    res += ")";
    return res;
}

namespace FJML {

/**
 * This namespace contains some linear algebra helper functions
 */
namespace LinAlg {

/**
 * @brief Compute the dot product of two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @return The dot product of the two vectors.
 */
template <typename T> inline double dot_product(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.data_size[0] != b.data_size[0]) {
        throw std::invalid_argument("The two vectors must have the same size.");
    }
    double res = 0;
    for (int i = 0; i < (int)a.data_size[0]; i++) {
        res += a.data[i] * b.data[i];
    }
    return res;
}

/**
 * @brief Multiplies two matrices.
 *
 * @param a The first matrix.
 * @param b The second matrix.
 * @return The product.
 */
template <typename T> inline Tensor<T> matrix_multiply(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.dim() == 1 && b.dim() == 2) {
        if (a.shape[0] != b.shape[0]) {
            throw std::invalid_argument("Invalid matrix dimensions: " + print_shape(a) + " and " + print_shape(b));
        }
        Tensor<T> result({b.shape[1]});
        for (int i = 0; i < b.shape[1]; i++) {
            for (int j = 0; j < b.shape[0]; j++) {
                result.data[i] += a.data[j] * b.data[j * b.shape[1] + i];
            }
        }
        return result;
    } else if (a.dim() == 2 && b.dim() == 1) {
        if (a.shape[1] != b.shape[0]) {
            throw std::invalid_argument("Invalid matrix dimensions: " + print_shape(a) + " and " + print_shape(b));
        }
        Tensor<T> result({a.shape[0]});
        for (int i = 0; i < a.shape[0]; i++) {
            for (int j = 0; j < a.shape[1]; j++) {
                result.data[i] += a.data[i * a.shape[1] + j] * b.data[j];
            }
        }
        return result;
    } else if (a.dim() != 2 || b.dim() != 2 || a.shape[1] != b.shape[0]) {
        throw std::invalid_argument("Invalid matrix dimensions: " + print_shape(a) + " and " + print_shape(b));
    }
    Tensor<T> result({a.shape[0], b.shape[1]});
    for (int i = 0; i < a.shape[0]; i++) {
        for (int k = 0; k < a.shape[1]; k++) {
            for (int j = 0; j < b.shape[1]; j++) {
                result.data[i * b.shape[1] + j] += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
            }
        }
    }
    return result;
}

/**
 * Transposes a matrix.
 * @param a The matrix.
 * @return The transpose of the matrix.
 */
template <typename T> inline Tensor<T> transpose(const Tensor<T>& a) {
    if (a.dim() != 2) {
        throw std::invalid_argument("Argument must be a matrix");
    }
    Tensor<T> result({a.shape[1], a.shape[0]});
    for (int i = 0; i < a.shape[0]; i++) {
        for (int j = 0; j < a.shape[1]; j++) {
            result.data[j * a.shape[0] + i] = a.data[i * a.shape[1] + j];
        }
    }
    return result;
}

/**
 * Sums all the elements in a tensor.
 *
 * @param a The tensor.
 * @return The sum of all the elements in the tensor.
 */
template <typename T> inline T sum(const Tensor<T>& a) {
    T res = 0;
    for (int i = 0; i < a.data_size[0]; i++) {
        res += a.data[i];
    }
    return res;
}

/**
 * Randomly chooses an index using tensor values as probabilities
 *
 * @param a the probabilities
 * @return a randomly chosen index
 */
template <typename T> inline int random_choice(const Tensor<T>& a) {
    T rand_num = (T)rand() / (T)RAND_MAX;
    for (int i = 0; i < a.data_size[0]; i++) {
        if (rand_num < a.data[i]) {
            return i;
        }
        rand_num -= a.data[i];
    }
    return a.data_size[0] - 1;
}

/**
 * Computes the index of the maximum value in a tensor.
 *
 * @param a The tensor.
 * @return The index of the maximum value in the tensor.
 */
template <typename T> inline int argmax(const Tensor<T>& a) {
    int res = 0;
    for (int i = 1; i < a.data_size[0]; i++) {
        if (a.data[i] > a.data[res]) {
            res = i;
        }
    }
    return res;
}

} // namespace LinAlg

} // namespace FJML

#endif
