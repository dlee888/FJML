// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <cstring>

#include "../include/FJML/data.h"

namespace FJML {

namespace Data {

Tensor one_hot(int x, int n) {
    std::vector<int> shape;
    shape.push_back(n);
    Tensor res(shape);
    res.at(x) = 1;
    return res;
}

void split(const Tensor& input_set, const Tensor& output_set, Tensor& input_train, Tensor& output_train,
           Tensor& input_test, Tensor& output_test, double train_frac) {
    int n = input_set.shape[0];
    int train_n = n * train_frac;

    std::vector<int> indices(n);
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }
    std::random_shuffle(indices.begin(), indices.end());

    input_train = Tensor({train_n, input_set.shape[1]}, input_set.device);
    output_train = Tensor({train_n, output_set.shape[1]}, output_set.device);
    input_test = Tensor({n - train_n, input_set.shape[1]}, input_set.device);
    output_test = Tensor({n - train_n, output_set.shape[1]}, output_set.device);

    for (int i = 0; i < train_n; i++) {
        std::memcpy(input_train.data + i * input_set.shape[1], input_set.data + indices[i] * input_set.shape[1],
                    input_set.shape[1] * sizeof(double));
        std::memcpy(output_train.data + i * output_set.shape[1], output_set.data + indices[i] * output_set.shape[1],
                    output_set.shape[1] * sizeof(double));
    }
    for (int i = train_n; i < n; i++) {
        std::memcpy(input_test.data + (i - train_n) * input_set.shape[1],
                    input_set.data + indices[i] * input_set.shape[1], input_set.shape[1] * sizeof(double));
        std::memcpy(output_test.data + (i - train_n) * output_set.shape[1],
                    output_set.data + indices[i] * output_set.shape[1], output_set.shape[1] * sizeof(double));
    }
}

} // namespace Data

} // namespace FJML
