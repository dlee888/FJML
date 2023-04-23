// Copyright (c) 2023 David Lee
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
 * If more than two dimensions are given, the inputs will be treated as an array of matrices.
 *
 * @param a The first matrix.
 * @param b The second matrix.
 * @return The product.
 */
template <typename T> inline Tensor<T> matrix_multiply(Tensor<T>& a, Tensor<T>& b) {
    if (a.dim() == 1 && b.dim() == 2) {
        if (a.shape[0] != b.shape[0]) {
            throw std::invalid_argument("Invalid matrix dimensions: " + print_shape(a) + " and " + print_shape(b));
        }
        a.reshape({1, a.shape[0]});
    } else if (a.dim() == 2 && b.dim() == 1) {
        if (a.shape[1] != b.shape[0]) {
            throw std::invalid_argument("Invalid matrix dimensions: " + print_shape(a) + " and " + print_shape(b));
        }
        b.reshape({b.shape[0], 1});
    } else if (a.dim() != b.dim() || a.dim() < 2 || a.shape[a.dim() - 1] != b.shape[b.dim() - 2]) {
        throw std::invalid_argument("Invalid matrix dimensions: " + print_shape(a) + " and " + print_shape(b));
    }
    std::vector<int> result_shape;
    for (int i = 0; i < a.dim() - 2; i++) {
        if (a.shape[i] != b.shape[i]) {
            throw std::invalid_argument("Invalid matrix dimensions: " + print_shape(a) + " and " + print_shape(b));
        }
        result_shape.push_back(a.shape[i]);
    }
    result_shape.push_back(a.shape[a.dim() - 2]);
    result_shape.push_back(b.shape[b.dim() - 1]);
    Tensor<T> result(result_shape);
    int rows = a.shape[a.dim() - 2];
    int cols = b.shape[b.dim() - 1];
    int inner = a.shape[a.dim() - 1];
    int stride = rows * cols;
    int a_stride = rows * inner;
    int b_stride = inner * cols;
    int num_matrices = a.data_size[0] / (rows * inner);
    for (int i = 0; i < num_matrices; i++) {
        for (int j = 0; j < rows; j++) {
            for (int l = 0; l < inner; l++) {
                for (int k = 0; k < cols; k++) {
                    result.data[i * stride + j * cols + k] += a.data[i * a_stride + j * inner + l] *
                                                              b.data[i * b_stride + l * cols + k];
                }
            }
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

} // namespace LinAlg

} // namespace FJML

#endif
