// Copyright (c) 2024 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <chrono>
#include <cmath>
#include <cstring>
#include <random>

#include "../include/FJML/layers.h"

namespace FJML {

namespace Layers {

ConvND::ConvND(std::vector<int> shape, int channels_in, int channels_out, std::vector<int> stride,
               Activations::Activation activ, Padding padding, std::vector<int> padding_size)
    : Layer{"ConvND"}, shape{shape}, channels_in{channels_in}, channels_out{channels_out}, stride{stride}, activ{activ},
      padding{padding}, padding_size{padding_size}, w_opt{nullptr}, b_opt{nullptr} {
    if (stride.size() == 0) {
        stride = std::vector<int>(shape.size(), 1);
    }
    if (padding_size.size() == 0) {
        padding = std::vector<int>(shape.size(), 0);
    }

    std::vector<int> filter_shape = shape;
    filter_shape.push_back(channels_in);
    filter_shape.push_back(channels_out);
    filters = Tensor(filter_shape);
    bias = Tensor(std::vector<int>{channels_out});
    // TODO: initialize weights
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::normal_distribution d(0.0, std::sqrt(2.0 / input));
    // for (int i = 0; i < input_size; i++) {
    //     for (int j = 0; j < output_size; j++) {
    //         weights.at(i, j) = d(gen);
    //     }
    // }
}

ConvND::ConvND(std::ifstream& file)
    : Layer{"ConvND"}, weights{{0}}, bias{{0}}, activ{Activations::Activation(
                                                    "", [](float x) { return x; }, [](float x) { return 1; })} {
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
    // TODO: Implement this
    // file >> input_size >> output_size;
    // if (input_size <= 0 || output_size <= 0) {
    //     throw std::runtime_error("Invalid input or output size for Dense layer");
    // }
    // weights = Tensor({input_size, output_size});
    // bias = Tensor({output_size});
    // for (int i = 0; i < input_size; i++) {
    //     for (int j = 0; j < output_size; j++) {
    //         file >> weights.at(i, j);
    //     }
    // }
    // for (int i = 0; i < output_size; i++) {
    //     file >> bias.at(i);
    // }
}

ConvND::~ConvND() {
    if (w_opt) {
        delete w_opt;
    }
    if (b_opt) {
        delete b_opt;
    }
}

Tensor ConvND::apply(const Tensor& input) const {
    // TODO: make this more efficient
    if (input.ndim() != shape.size() + 1 || input.shape[input.ndim() - 1] != channels_in) {
        throw std::invalid_argument("The shape of the input is incorrect");
    }

    std::vector<int> output_shape;
    for (int i = 0; i < input.ndim() - 1; i++) {
        output_shape.append((input.shape[i] + 2 * padding_size[i] - shape[i] + 1) / stride[i]);
    }
    Tensor output(output_shape);
}

Tensor ConvND::backward(const Tensor& input_vals, const Tensor& output_grad) {
    // TODO: implement this
    // int n = input_vals.shape[0];
    //
    // Tensor activ_grad = activ.backward(LinAlg::dense_forward(input_vals, weights, bias)) * output_grad;
    //
    // Tensor w_grad = LinAlg::matrix_multiply(LinAlg::transpose(input_vals), activ_grad);
    // Tensor b_grad = Tensor({output_size}, activ_grad.device);
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < output_size; j++) {
    //         b_grad.data[j] += activ_grad.data[i * output_size + j];
    //     }
    // }
    //
    // Tensor prev_grad = LinAlg::matrix_multiply(activ_grad, LinAlg::transpose(weights));
    //
    // w_grad /= n;
    // b_grad /= n;
    // w_opt->apply_grad(weights, w_grad);
    // b_opt->apply_grad(bias, b_grad);
    //
    // return prev_grad;
}

void ConvND::save(std::ofstream& file) const {
    // TODO: implement this
    // file << "Dense" << std::endl;
    // file << activ.name << std::endl;
    // file << input_size << " " << output_size << " ";
    // for (int i = 0; i < input_size; i++) {
    //     for (int j = 0; j < output_size; j++) {
    //         file << weights.at(i, j) << " ";
    //     }
    // }
    // for (int i = 0; i < output_size; i++) {
    //     file << bias.at(i) << " ";
    // }
    // file << std::endl;
}

void ConvND::summary() const {
    // TODO: implement this
    // std::cout << "Dense layer with " << input_size << " inputs and " << output_size << " outputs" << std::endl;
    // std::cout << "Activation function: " << activ.name << std::endl;
}

void ConvND::set_optimizer(const Optimizers::Optimizer* opt) {
    delete w_opt;
    delete b_opt;
    w_opt = opt->clone();
    b_opt = opt->clone();
}

} // namespace Layers

} // namespace FJML
