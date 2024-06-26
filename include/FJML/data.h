// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef DATA_INCLUDED
#define DATA_INCLUDED

#include <algorithm>

#include "tensor.h"

namespace FJML {

/**
 * @brief Helper functions for loading data
 */
namespace Data {

/**
 * @brief One hot encoding of a Tensor x
 *
 * Assumes x is a tensor of integers. If x has shape (a, b, ...), the output will have shape (a, b, ..., n)
 *
 * @param x The number to encode
 * @param n The total number of possible values
 * @return The one hot encoding of x
 */
Tensor one_hot(Tensor x, int n);

/**
 * @brief Split data into training and testing sets
 *
 * The data should be given as a matrix, where each row is a data point.
 *
 * @param input_set The input data
 * @param output_set The output data
 * @param input_train The vector where the training input data will be stored
 * @param output_train The vector where the training output data will be stored
 * @param input_test The vector where the testing input data will be stored
 * @param output_test The vector where the testing output data will be stored
 * @param train_frac The percentage of the data to be used for training (from 0.0 - 1.0)
 */
void split(const Tensor& input_set, const Tensor& output_set, Tensor& input_train, Tensor& output_train,
           Tensor& input_test, Tensor& output_test, float train_frac = 0.8);

} // namespace Data

} // namespace FJML

#endif
