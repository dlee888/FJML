// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include "../include/FJML/data.h"
#include <iostream>

namespace FJML {

namespace Data {

Tensor one_hot(int x, int n) {
    // std::cerr << "One hot encoding " << x << " " << n << std::endl;
    std::vector<int> shape;
    shape.push_back(n);
    // std::cerr << "Shape: " << shape[0] << std::endl;
    // std::cerr << "Creating tensor" << std::endl;
    Tensor res(shape);
    res.at(x) = 1;
    return res;
}

void split(const std::vector<Tensor>& input_set, const std::vector<Tensor>& output_set,
           std::vector<Tensor>& input_train, std::vector<Tensor>& output_train, std::vector<Tensor>& input_test,
           std::vector<Tensor>& output_test, double train_frac) {
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
