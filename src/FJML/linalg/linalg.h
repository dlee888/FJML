// Copyright (c) 2022 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef LINALG_INCLUDED
#define LINALG_INCLUDED

#include <cassert>
#include <random>
#include <stdexcept>

#include "tensor.h"

/**
 * @brief Helper method to print the shape of a tensor.
 * @param a The tensor.
 * @return The shape of the tensor.
 */
template <typename T> static std::string print_shape(const FJML::Tensor<T>& a) {
    std::string res = "(";
    for (int i = 0; i < a.dims(); i++) {
        res += std::to_string(a.shape[i]) + ", ";
    }
    res.pop_back();
    res.pop_back();
    res += ")";
    return res;
}

namespace FJML {

namespace LinAlg {

/**
 * @brief Compute the dot product of two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @return The dot product of the two vectors.
 */
template <typename T> inline double dotProduct(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.data_size[0] != b.data_size[0]) {
        throw std::invalid_argument("The two vectors must have the same size.");
    }
    double res = 0;
    for (int i = 0; i < (int)a.size(); i++) {
        res += a.data[i] * b.data[i];
    }
    return res;
}

/**
 * @brief Multiplies two matrices.
 * @param a The first matrix.
 * @param b The second matrix.
 * @return The product.
 */
template <typename T> inline Tensor<T> matrixMultiply(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.dims() == 1 && b.dims() == 2) {
        if (a.shape[0] != b.shape[0]) {
            throw std::invalid_argument("Invalid matrix dimensions: " + print_shape(a) + " and " + print_shape(b));
        }
        a.reshape(1, a.shape[0]);
    } else if (a.dims() == 2 && b.dims() == 1) {
        if (a.shape[1] != b.shape[0]) {
            throw std::invalid_argument("Invalid matrix dimensions: " + print_shape(a) + " and " + print_shape(b));
        }
        b.reshape(1, b.shape[0]);
    } else if (a.dims() != b.dims() || a.dims() < 2 || a.shape[a.dims() - 1] != b.shape[b.dims() - 2]) {
        throw std::invalid_argument("Invalid matrix dimensions: " + print_shape(a) + " and " + print_shape(b));
    }
    std::vector<int> result_shape;
    for (int i = 0; i < a.dims() - 1; i++) {
        result_shape.push_back(a.shape[i]);
    }
    result_shape.push_back(b.shape[b.dims() - 1]);
    Tensor<T> res(result_shape);
    int i, j, k;
#pragma omp parallel for private(i, j, k) shared(a, b, res)
    for (i = 0; i < (int)a.data_size[0]; i++) {
        for (j = 0; j < (int)b.data_size[1]; j++) {
            for (k = 0; k < (int)a.data_size[1]; k++) {
                res.data[i * b.data_size[1] + j] += a.data[i * a.data_size[1] + k] * b.data[k * b.data_size[1] + j];
            }
        }
    }
    return res;
}

/**
 * @brief Multiplies a matrix by a vector.
 * @param a The matrix.
 * @param b The vector.
 * @return The product of the matrix and the vector.
 */
inline Tensor<T> matrixMultiply(const weights& w, const Tensor<T>& a) {
    assert(a.size() == w[0].size());
    Tensor<T> res((int)w.size());
    int i, j;
#pragma omp parallel for private(i, j) shared(w, a)
    for (i = 0; i < (int)w.size(); i++) {
        for (j = 0; j < (int)a.size(); j++) {
            res[i] += w[i][j] * a[j];
        }
    }
    return res;
}

/**
 * @brief Takes the largest value in a vector.
 * @param a The vector.
 * @return The index of the largest value in the vector.
 */
inline int argmax(const Tensor<T>& a) {
    int res = 0;
    for (int i = 1; i < (int)a.size(); i++) {
        if (a[i] > a[res]) {
            res = i;
        }
    }
    return res;
}

template <int N> inline double sum(const Tensor<N>& a) {
    double res = 0;
    for (int i = 0; i < a.size(); i++) {
        res += sum(a[i]);
    }
    return res;
}

template <> inline double sum(const Tensor<1>& a) {
    double res = 0;
    for (int i = 0; i < a.size(); i++) {
        res += a[i];
    }
    return res;
}

inline int random_choice(const Tensor<1>& a) {
    double rand_num = (double)rand() / RAND_MAX;
    for (int i = 0; i < a.size(); i++) {
        if (rand_num < a[i]) {
            return i;
        }
        rand_num -= a[i];
    }
    return a.size() - 1;
}

} // namespace LinAlg

} // namespace FJML

#endif
