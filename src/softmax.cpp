// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <cmath>
#include <iostream>

#include "../include/FJML/layers.h"

namespace FJML {

namespace Layers {

Tensor Softmax::norm(const Tensor& input) const {
    Tensor ret(input);
    double max = input.at(0);
    for (auto& i : ret) {
        max = std::max(max, i);
    }
    for (auto& i : ret) {
        i -= max;
    }
    return ret;
}

Tensor Softmax::apply(const Tensor& input) const {
    Tensor res = norm(input);
    double sum = 0;

    for (double& d : res) {
        d = exp(d);
        sum += d;
    }
    for (double& d : res) {
        d /= sum;
    }

    return res;
}

std::vector<Tensor> Softmax::apply(const std::vector<Tensor>& input) const {
    std::vector<Tensor> res;
    for (const Tensor& l : input) {
        res.push_back(apply(l));
    }
    return res;
}

std::vector<Tensor> Softmax::backward(const std::vector<Tensor>& input_vals, const std::vector<Tensor>& output_grad) {
    int n = input_vals.size(), m = input_vals[0].data_size[0];

    std::vector<Tensor> res(n, Tensor{input_vals[0].shape});
    for (int i = 0; i < n; i++) {
        Tensor out = norm(input_vals[i]);

        double sum = 0;
        for (double& d : out) {
            d = exp(d);
            sum += d;
        }

        double denom = sum * sum;
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < m; k++) {
                res[i].data[j] += output_grad[i].at(k) * (out.at(j) * (k == j ? sum - out.at(j) : -out.at(k))) / denom;
            }
        }
    }

    return res;
}

void Softmax::save(std::ofstream& file) const { file << "Softmax" << std::endl; }

void Softmax::summary() const { std::cout << "Softmax layer" << std::endl; }

} // namespace Layers

} // namespace FJML
