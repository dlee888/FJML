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

#include "tensor.h"

namespace FJML {

/**
 * @brief This namespace contains some linear algebra helper functions
 */
namespace LinAlg {

/**
 * @brief Compute the dot product of two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @return The dot product of the two vectors.
 */
float dot_product(const Tensor& a, const Tensor& b);

/**
 * @brief Multiplies two matrices.
 *
 * If the tensors are vectors, the first will be interpreted as a column vector, and the second will be interpreted as a
 * row vector.
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
 * @brief Transposes a matrix.
 * @param a The matrix.
 * @return The transpose of the matrix.
 */
Tensor transpose(const Tensor& a);

/**
 * @brief Sums all the elements in a tensor.
 *
 * @param a The tensor.
 * @return The sum of all the elements in the tensor.
 */
float sum(const Tensor& a);

/**
 * @brief Computes the mean of all the elements in a tensor.
 *
 * @param a The tensor.
 * @return The mean of all the elements in the tensor.
 */
float mean(const Tensor& a);

/**
 * @brief Raises all the elements in a tensor to a power.
 * @param a The tensor.
 * @param b The power.
 * @return The tensor with all the elements raised to the power.
 */
Tensor pow(const Tensor& a, float b);

/**
 * @brief Randomly chooses an index using tensor values as probabilities
 *
 * @param a the probabilities
 * @return a randomly chosen index
 */
int random_choice(const Tensor& a);

/**
 * @brief Computes the maximum value in a tensor.
 * @param a The tensor.
 * @return The maximum value in the tensor.
 */
float max(const Tensor& a);

/**
 * @brief Computes the index of the maximum value in a tensor, given an axis to compute along.
 *
 * For example, if the input is a matrix and the axis is 0, the output will be a vector containing the maximum of each
 * column. If the input is a matrix and the axis is 1, the output will be a vector containing the maximum of each row.
 *
 * The default axis is -1, which means the last axis.
 *
 * @param a The tensor.
 * @param axis The axis to compute the maximum value along.
 * @return A tensor containing the indices of the maximum value along the specified axis.
 */
Tensor argmax(const Tensor& a, int axis = -1);

/**
 * @brief Returns a tensor containing one where the two tensors are equal, and zero otherwise.
 * @param a The first tensor.
 * @param b The second tensor.
 * @return A tensor containing one where the two tensors are equal, and zero otherwise.
 */
Tensor equal(const Tensor& a, const Tensor& b);

/**
 * @brief Forward pass of a dense layer.
 * @param input The input tensor.
 * @param weights The weights tensor.
 * @param bias The bias tensor.
 * @return The output tensor.
 */
Tensor dense_forward(const Tensor& input, const Tensor& weights, const Tensor& bias);

} // namespace LinAlg

} // namespace FJML

#endif
