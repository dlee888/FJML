// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <cmath>
#include <iostream>

#include "../include/FJML/layers.h"

namespace FJML {

namespace Layers {

Tensor Softmax::apply(const Tensor& input) const {
    Tensor res = input;
    for (int i = 0; i < res.shape[0]; i++) {
        float max = res.data[0];
        for (int j = 1; j < res.shape[1]; j++) {
            if (res.data[i * res.shape[1] + j] > max) {
                max = res.data[i * res.shape[1] + j];
            }
        }
        float sum = 0;
        for (int j = 0; j < res.shape[1]; j++) {
            res.data[i * res.shape[1] + j] = exp(res.data[i * res.shape[1] + j] - max);
            sum += res.data[i * res.shape[1] + j];
        }
        for (int j = 0; j < res.shape[1]; j++) {
            res.data[i * res.shape[1] + j] /= sum;
        }
    }
    return res;
}

Tensor Softmax::backward(const Tensor& input_vals, const Tensor& output_grad) {
    Tensor res{input_vals.shape, 0.0, input_vals.device};
    Tensor out = input_vals;
    for (int i = 0; i < res.shape[0]; i++) {
        float max = out.data[i * res.shape[1]];
        for (int j = 1; j < res.shape[1]; j++) {
            if (out.data[i * res.shape[1] + j] > max) {
                max = out.data[i * res.shape[1] + j];
            }
        }
        float sum = 0;
        for (int j = 0; j < res.shape[1]; j++) {
            out.data[i * out.shape[1] + j] = exp(out.data[i * out.shape[1] + j] - max);
            sum += out.data[i * out.shape[1] + j];
        }

        float denom = sum * sum;
        for (int j = 0; j < res.shape[1]; j++) {
            for (int k = 0; k < res.shape[1]; k++) {
                if (j == k) {
                    res.data[i * res.shape[1] + j] += (sum - out.data[i * res.shape[1] + k]) / denom *
                                                      output_grad.data[i * res.shape[1] + k] *
                                                      out.data[i * res.shape[1] + k];
                } else {
                    res.data[i * res.shape[1] + j] += -out.data[i * res.shape[1] + k] / denom *
                                                      output_grad.data[i * res.shape[1] + k] *
                                                      out.data[i * res.shape[1] + j];
                }
            }
        }
    }

    return res;
}

void Softmax::save(std::ofstream& file) const { file << "Softmax" << std::endl; }

void Softmax::summary() const { std::cout << "Softmax layer" << std::endl; }

} // namespace Layers

} // namespace FJML
