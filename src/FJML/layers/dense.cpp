// Copyright (c) 2022 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

#include "layers.h"

namespace FJML {

namespace Layers {

Layers::Dense::Dense(int input, int output, Activations::Activation activ, Optimizers::Optimizer* opt, bool randomize)
    : Layer{"Dense"}, input_size{input}, output_size{output}, activ{activ}, w_opt{opt->clone()}, b_opt{opt->clone()},
      weights{Tensor<double>(std::vector<int>{input, output})}, bias{Tensor<double>(std::vector<int>{output})} {
    if (randomize) {
        unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937_64 rng = std::mt19937_64(seed);
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < output_size; j++) {
                weights.at(i, j) = (double)rng() / (double)(rng.max() - rng.min()) * 2 - 1;
            }
        }
        for (int i = 0; i < output_size; i++) {
            bias.at(i) = (double)rng() / (double)(rng.max() - rng.min()) * 2 - 1;
        }
    }
}

Tensor<double> Layers::Dense::apply(const Tensor<double>& input) const {
    return LinAlg::matrix_multiply(input, weights) + bias;
}

std::vector<Tensor<double>> Layers::Dense::apply(const std::vector<Tensor<double>>& input) const {
    std::vector<Tensor<double>> res;
    for (const Tensor<double>& l : input) {
        res.push_back(apply(l));
    }
    return res;
}

std::vector<Tensor<double>> Layers::Dense::backward(const std::vector<Tensor<double>>& input_vals,
                                                    const std::vector<Tensor<double>>& output_vals,
                                                    const std::vector<Tensor<double>>& output_grad) {
    int n = input_vals.size();

    Tensor<double> w_grad({input_size, output_size});
    Tensor<double> b_grad({output_size});
    std::vector<Tensor<double>> prev_grad(n, Tensor<double>({input_size}));

    for (int datapoint = 0; datapoint < n; datapoint++) {
        Tensor<double> out_grad = activ.backward(output_vals[datapoint]) * output_grad[datapoint];

        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < output_size; j++) {
                w_grad.at(i, j) += input_vals[datapoint].at(i) * out_grad.at(j);
                prev_grad[datapoint].at(i) += weights.at(i, j) * out_grad.at(j);
            }
        }

        b_grad += out_grad;
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
    : Layer{"Dense"}, activ{Activations::Activation(
                          "", [](double x) { return x; }, [](double x) { return 1; })},
      bias{{0}}, weights{{0}} {
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
    weights = Tensor<double>({input_size, output_size});
    bias = Tensor<double>({output_size});
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
