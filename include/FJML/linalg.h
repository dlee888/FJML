// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef LINALG_INCLUDED
#define LINALG_INCLUDED

#include <cstdlib>
#include <iostream>
#include <random>

#ifdef CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#pragma GCC target("avx2,fma")
#pragma GCC optimize("O3,unroll-loops")

#include "tensor.h"

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
double dot_product(const Tensor& a, const Tensor& b);

/**
 * @brief Multiplies two matrices.
 *
 * If the tensors are vectors, the first will be interpreted as a column vector, and the second will be interpreted as a row vector.
 *
 * If `a` is a vector and `b` is a matrix, `a` will be interpreted as a row vector
 * If `b` is a vector and `a` is a matrix, `b` will be interpreted as a column vector
 *
 * @param a The first matrix.
 * @param b The second matrix.
 * @return The product.
 */
Tensor matrix_multiply(const Tensor& a, const Tensor& b);

/**
 * Transposes a matrix.
 * @param a The matrix.
 * @return The transpose of the matrix.
 */
Tensor transpose(const Tensor& a);

/**
 * Sums all the elements in a tensor.
 *
 * @param a The tensor.
 * @return The sum of all the elements in the tensor.
 */
double sum(const Tensor& a);

/**
 * Randomly chooses an index using tensor values as probabilities
 *
 * @param a the probabilities
 * @return a randomly chosen index
 */
int random_choice(const Tensor& a);

/**
 * Computes the index of the maximum value in a tensor.
 *
 * @param a The tensor.
 * @return The index of the maximum value in the tensor.
 */
int argmax(const Tensor& a);

/**
 * Forward pass of a dense layer.
 * @param input The input tensor.
 * @param weights The weights tensor.
 * @param bias The bias tensor.
 * @return The output tensor.
 */
Tensor dense_forward(const Tensor& input, const Tensor& weights, const Tensor& bias);

} // namespace LinAlg

} // namespace FJML

#endif
