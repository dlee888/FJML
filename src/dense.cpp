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

Layers::Dense::Dense(int input, int output, Activations::Activation activ, Device device)
    : Layer{"Dense"}, input_size{input}, output_size{output}, weights{Tensor(std::vector<int>{input, output}, device)},
      bias{Tensor(std::vector<int>{output}, device)}, activ{activ}, w_opt{nullptr}, b_opt{nullptr} {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution d(0.0, std::sqrt(2.0 / input));
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < output_size; j++) {
            weights.at(i, j) = d(gen);
        }
    }
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

Dense::~Dense() {
    delete w_opt;
    delete b_opt;
}

Tensor Layers::Dense::apply(const Tensor& input) const {
    Tensor res = LinAlg::dense_forward(input, weights, bias);
    return activ.apply(res);
}

Tensor Layers::Dense::backward(const Tensor& input_vals, const Tensor& output_grad) {
    int n = input_vals.shape[0];

    Tensor activ_grad = activ.backward(LinAlg::dense_forward(input_vals, weights, bias)) * output_grad;
    // Tensor activ_grad = LinAlg::dense_forward(input_vals, weights, bias);
    // activ.apply_derivative(activ_grad);
    // activ_grad *= output_grad;

    Tensor w_grad = LinAlg::matrix_multiply(LinAlg::transpose(input_vals), activ_grad);
    Tensor b_grad = Tensor({output_size}, activ_grad.device);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < output_size; j++) {
            b_grad.data[j] += activ_grad.data[i * output_size + j];
        }
    }

    Tensor prev_grad = LinAlg::matrix_multiply(activ_grad, LinAlg::transpose(weights));

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

void Layers::Dense::summary() const {
    std::cout << "Dense layer with " << input_size << " inputs and " << output_size << " outputs" << std::endl;
    std::cout << "Activation function: " << activ.name << std::endl;
}

void Layers::Dense::set_optimizer(const Optimizers::Optimizer* opt) {
    delete w_opt;
    delete b_opt;
    w_opt = opt->clone();
    b_opt = opt->clone();
}

} // namespace Layers

} // namespace FJML
