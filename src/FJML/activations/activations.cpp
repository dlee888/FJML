// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <cmath>

#include "activations.h"

namespace FJML {

namespace Activations {

const Activation sigmoid = Activation(
    "sigmoid", [](double x) { return 1 / (1 + std::exp(-x)); },
    [](double x) { return std::exp(-x) / std::pow(1 + std::exp(-x), 2); });

const Activation tanh = Activation(
    "tanh", [](double x) { return std::tanh(x); }, [](double x) { return 1 - std::pow(std::tanh(x), 2); });

const Activation relu = Activation(
    "relu", [](double x) { return x > 0 ? x : 0; }, [](double x) { return x > 0 ? 1 : 0; });

const Activation leaky_relu = Activation(
    "leaky relu", [](double x) { return x > 0 ? x : 0.01 * x; }, [](double x) { return x > 0 ? 1 : 0.01; });

const Activation linear = Activation(
    "linear", [](double x) { return x; }, [](double x) { return 1; });

const Activation swish = Activation(
    "swish", [](double x) { return x / (1 + std::exp(-x)); },
    [](double x) {
        double exp = std::exp(x);
        double exp_plus_one = 1 + exp;
        return 1 + (x - 1) / exp_plus_one - x / exp_plus_one / exp_plus_one;
    });

} // namespace Activations

} // namespace FJML
