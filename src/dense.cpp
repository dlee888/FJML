// Copyright (c) 2022 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

#include "../include/FJML/layers.h"

namespace FJML {

namespace Layers {

Layers::Dense::Dense(int input, int output, Activations::Activation activ, Optimizers::Optimizer* opt, bool randomize)
    : Layer{"Dense"}, input_size{input}, output_size{output}, weights{Tensor(std::vector<int>{input, output})},
      bias{Tensor(std::vector<int>{output})}, activ{activ}, w_opt{opt->clone()}, b_opt{opt->clone()} {
    if (randomize) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution d(0.0, std::sqrt(2.0 / input));
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < output_size; j++) {
                weights.at(i, j) = d(gen);
            }
        }
    }
}

Tensor Layers::Dense::apply(const Tensor& input) const { return LinAlg::matrix_multiply(input, weights) + bias; }

std::vector<Tensor> Layers::Dense::apply(const std::vector<Tensor>& input) const {
    std::vector<Tensor> res;
    for (const Tensor& l : input) {
        res.push_back(apply(l));
    }
    return res;
}

std::vector<Tensor> Layers::Dense::backward(const std::vector<Tensor>& input_vals,
                                            const std::vector<Tensor>& output_grad) {
    int n = input_vals.size();

    Tensor w_grad({input_size, output_size});
    Tensor b_grad({output_size});
    std::vector<Tensor> prev_grad(n, Tensor({input_size}));

    for (int datapoint = 0; datapoint < n; datapoint++) {
        Tensor activ_grad = activ.backward(LinAlg::matrix_multiply(input_vals[datapoint], weights) + bias) *
                            output_grad[datapoint];

        w_grad += LinAlg::matrix_multiply(input_vals[datapoint], activ_grad);
        b_grad += activ_grad;
        prev_grad[datapoint] = LinAlg::matrix_multiply(weights, activ_grad);
    }

    w_grad /= n;
    b_grad /= n;
    w_opt->apply_grad(weights, w_grad);
    b_opt->apply_grad(bias, b_grad);

    return prev_grad;
}

void Layers::Dense::save(std::ofstream& file) const {
    file << "Dense" << std::endl;
    file << activ.name << std::endl;
    file << input_size << " " << output_size << " ";
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < output_size; j++) {
            file << weights.at(i, j) << " ";
        }
    }
    for (int i = 0; i < output_size; i++) {
        file << bias.at(i) << " ";
    }
    file << std::endl;
}

Layers::Dense::Dense(std::ifstream& file)
    : Layer{"Dense"}, weights{{0}}, bias{{0}}, activ{Activations::Activation(
                                                   "", [](double x) { return x; }, [](double x) { return 1; })} {
    std::string activation;
    file >> activation;
    for (Activations::Activation a : Activations::activations) {
        if (a.name == activation) {
            activ = a;
            break;
        }
    }
    if (activ.name != activation) {
        throw std::runtime_error("Unknown activation function");
    }
    file >> input_size >> output_size;
    weights = Tensor({input_size, output_size});
    bias = Tensor({output_size});
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < output_size; j++) {
            file >> weights.at(i, j);
        }
    }
    for (int i = 0; i < output_size; i++) {
        file >> bias.at(i);
    }
}

void Layers::Dense::summary() const {
    std::cout << "Dense layer with " << input_size << " inputs and " << output_size << " outputs" << std::endl;
    std::cout << "Activation function: " << activ.name << std::endl;
}

void Layers::Dense::set_optimizer(const Optimizers::Optimizer* opt) {
    w_opt = opt->clone();
    b_opt = opt->clone();
}

} // namespace Layers

} // namespace FJML
