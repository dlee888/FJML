// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef DATA_INCLUDED
#define DATA_INCLUDED

#include <algorithm>

#include "../linalg/tensor.h"

namespace FJML {

/**
 * @brief Helper functions for loading data
 */
namespace Data {

/**
 * @brief One hot encoding of a number x
 * @param x The number to encode
 * @param n The total number of possible values
 * @return The one hot encoding of x
 */
inline Tensor<double> one_hot(int x, int n) {
    Tensor<double> res({n});
    res.at(x) = 1;
    return res;
}

/**
 * @brief Split data into training and testing sets
 *
 * @param input_set The input data
 * @param output_set The output data
 * @param input_train The vector where the training input data will be stored
 * @param output_train The vector where the training output data will be stored
 * @param input_test The vector where the testing input data will be stored
 * @param output_test The vector where the testing output data will be stored
 * @param train_frac The percentage of the data to be used for training (from 0.0 - 1.0)
 * @tparam T The type of the data
 */
template <typename T>
inline void split(const std::vector<Tensor<T>>& input_set, const std::vector<Tensor<T>>& output_set,
                  std::vector<Tensor<T>>& input_train, std::vector<Tensor<T>>& output_train,
                  std::vector<Tensor<T>>& input_test, std::vector<Tensor<T>>& output_test, double train_frac = 0.8) {
    int n = input_set.size();
    int train_n = n * train_frac;

    std::vector<int> indices(n);
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }
    std::random_shuffle(indices.begin(), indices.end());

    input_train.clear();
    output_train.clear();
    input_test.clear();
    output_test.clear();
    for (int i = 0; i < train_n; i++) {
        input_train.push_back(input_set[indices[i]]);
        output_train.push_back(output_set[indices[i]]);
    }
    for (int i = train_n; i < n; i++) {
        input_test.push_back(input_set[indices[i]]);
        output_test.push_back(output_set[indices[i]]);
    }
}

} // namespace Data

} // namespace FJML

#endif
