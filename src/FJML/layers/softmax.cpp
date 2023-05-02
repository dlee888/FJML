// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <cmath>
#include <iostream>

#include "../../../include/FJML/layers.h"

namespace FJML {

namespace Layers {

Tensor<double> Softmax::norm(const Tensor<double>& input) const {
    Tensor<double> ret(input);
    double max = input.at(0);
    for (auto& i : ret) {
        max = std::max(max, i);
    }
    for (auto& i : ret) {
        i -= max;
    }
    return ret;
}

Tensor<double> Softmax::apply(const Tensor<double>& input) const {
    Tensor<double> res = norm(input);
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

std::vector<Tensor<double>> Softmax::apply(const std::vector<Tensor<double>>& input) const {
    std::vector<Tensor<double>> res;
    for (const Tensor<double>& l : input) {
        res.push_back(apply(l));
    }
    return res;
}

std::vector<Tensor<double>> Softmax::backward(const std::vector<Tensor<double>>& input_vals,
                                              const std::vector<Tensor<double>>& output_vals,
                                              const std::vector<Tensor<double>>& output_grad) {
    int n = input_vals.size(), m = input_vals[0].data_size[0];

    std::vector<Tensor<double>> res(n, Tensor<double>{input_vals[0].shape});
    for (int i = 0; i < n; i++) {
        Tensor<double> out = norm(input_vals[i]);

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
