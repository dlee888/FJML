// Copyright (c) 2022 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <cmath>

#include "layers.h"

namespace FJML {

namespace Layers {

Layers::Dense::Dense() {
    activ = Activations::sigmoid;
    input_size = 0;
    output_size = 0;
    name = "Dense";
}

Layers::Dense::Dense(int _input, int _output, Activations::Activation _activ, Optimizers::Optimizer<1>* _opt,
                     bool randomize) {
    input_size = _input;
    output_size = _output;
    layer_weights = weights(std::vector<int>{input_size, output_size});
    layer_bias = bias(output_size);
    activ = _activ;
    w_opt = Optimizers::get_optimizer<2>(_opt);
    b_opt = Optimizers::get_optimizer<1>(_opt);
    name = "Dense";
    if (randomize) {
        unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937_64 rng = std::mt19937_64(seed);
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < output_size; j++) {
                layer_weights[i][j] = (double)rng() / (double)(rng.max() - rng.min()) * 2 - 1;
            }
        }
        for (int i = 0; i < output_size; i++) {
            layer_bias[i] = (double)rng() / (double)(rng.max() - rng.min()) * 2 - 1;
        }
    }
}

layer_vals Layers::Dense::apply(const layer_vals& input) const {
    layer_vals res = LinAlg::matrixMultiply(input, layer_weights);
    res += layer_bias;
    activ.apply(res);
    return res;
}

std::vector<layer_vals> Layers::Dense::apply(const std::vector<layer_vals>& input) const {
    std::vector<layer_vals> res;
    for (const layer_vals& l : input) {
        res.push_back(apply(l));
    }
    return res;
}

std::vector<layer_vals> Layers::Dense::apply_grad(const std::vector<layer_vals>& input_vals,
                                                  const std::vector<layer_vals>& output_vals,
                                                  const std::vector<layer_vals>& output_grad) {
    assert(input_vals.size() == output_grad.size());
    assert((int)input_vals[0].size() == input_size);
    assert((int)output_grad[0].size() == output_size);

    int n = input_vals.size();

    weights w_grad = weights(std::vector<int>{input_size, output_size});
    bias b_grad = bias{output_size};
    std::vector<layer_vals> prev_grad(n, layer_vals{input_size});

    for (int datapoint = 0; datapoint < n; datapoint++) {
        layer_vals out_grad2{output_size}; // Save results to speed up
        for (int i = 0; i < output_size; i++) {
            out_grad2[i] = activ.grad(output_vals[datapoint][i]) * output_grad[datapoint][i];
            assert(!std::isnan(out_grad2[i]));
        }

        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < output_size; j++) {
                w_grad[i][j] += input_vals[datapoint][i] * out_grad2[j];
                assert(!std::isnan(w_grad[i][j]));
                prev_grad[datapoint][i] += layer_weights[i][j] * out_grad2[j];
                assert(!std::isnan(prev_grad[datapoint][i]));
            }
        }
        for (int i = 0; i < output_size; i++) {
            b_grad[i] += out_grad2[i];
            assert(!std::isnan(b_grad[i]));
        }
    }

    w_grad /= n;
    b_grad /= n;
    w_opt->apply_grad(layer_weights, w_grad);
    b_opt->apply_grad(layer_bias, b_grad);

    return prev_grad;
}

void Layers::Dense::save(std::ofstream& file) const {
    file << "Dense" << std::endl;
    file << activ.name << std::endl;
    file << input_size << " " << output_size << " ";
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < output_size; j++) {
            file << layer_weights[i][j] << " ";
        }
    }
    for (int i = 0; i < output_size; i++) {
        file << layer_bias[i] << " ";
    }
    file << std::endl;
}

Layers::Dense::Dense(std::ifstream& file) {
    name = "Dense";
    std::string activation;
    file >> activation;
    for (Activations::Activation a : Activations::all_activations) {
        if (a.name == activation) {
            activ = a;
            break;
        }
    }
    if (activ.name != activation) {
        throw std::runtime_error("Unknown activation function");
    }
    file >> input_size >> output_size;
    layer_weights = weights(std::vector<int>{input_size, output_size});
    layer_bias = bias{output_size};
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < output_size; j++) {
            file >> layer_weights[i][j];
        }
    }
    for (int i = 0; i < output_size; i++) {
        file >> layer_bias[i];
    }
}

void Layers::Dense::summary() const {
    std::cout << "Dense layer with " << input_size << " inputs and " << output_size << " outputs" << std::endl;
    std::cout << "Activation function: " << activ.name << std::endl;
}

} // namespace Layers

} // namespace FJML
