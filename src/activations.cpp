// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <cmath>

#include "../include/FJML/activations.h"

namespace FJML {

namespace Activations {

Activation::Activation(std::string name, std::function<float(float)> func, std::function<float(float)> derivative)
    : name{name}, func{func}, derivative{derivative} {}

Tensor Activation::apply(Tensor& layer) const { return layer.apply_function(func); }

Tensor Activation::apply_derivative(Tensor& layer) const { return layer.apply_function(derivative); }

Tensor Activation::forward(const Tensor& layer) const { return layer.calc_function(func); }

Tensor Activation::backward(const Tensor& layer) const { return layer.calc_function(derivative); }

/**
 * The sigmoid function.
 *
 * Equation:
 * \f[
 *   \sigma(x) = \frac{1}{1 + e^{-x}}
 * \f]
 */
const Activation sigmoid = Activation(
    "sigmoid", [](float x) { return 1 / (1 + std::exp(-x)); },
    [](float x) { return std::exp(-x) / std::pow(1 + std::exp(-x), 2); });

/**
 * The hyperbolic tangent function.
 *
 * Equation:
 * \f[
 *  \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
 * \f]
 */
const Activation tanh = Activation(
    "tanh", [](float x) { return std::tanh(x); }, [](float x) { return 1 - std::pow(std::tanh(x), 2); });

/**
 * The rectified linear unit function.
 *
 * Equation:
 * \f[
 * \text{ReLU}(x) = \begin{cases}
 *  x & \text{if } x > 0 \\
 *  0 & \text{otherwise}
 *  \end{cases}
 *  \f]
 */
const Activation relu = Activation(
    "relu", [](float x) { return x > 0 ? x : 0; }, [](float x) { return x > 0 ? 1 : 0; });

/**
 * The leaky rectified linear unit function.
 *
 * Equation:
 * \f[
 * \text{LeakyReLU}(x) = \begin{cases}
 * x & \text{if } x > 0 \\
 * 0.01x & \text{otherwise}
 * \end{cases}
 * \f]
 */
const Activation leaky_relu = Activation(
    "leaky relu", [](float x) { return x > 0 ? x : 0.01 * x; }, [](float x) { return x > 0 ? 1 : 0.01; });

/**
 * The linear function.
 *
 * Equation:
 * \f[
 * \text{linear}(x) = x
 * \f]
 */
const Activation linear = Activation(
    "linear", [](float x) { return x; }, [](float x) { return 1; });

/**
 * The swish function.
 *
 * Equation:
 * \f[
 * \text{swish}(x) = \frac{x}{1 + e^{-x}}
 * \f]
 */
const Activation swish = Activation(
    "swish", [](float x) { return x / (1 + std::exp(-x)); },
    [](float x) {
        float exp = std::exp(x);
        float exp_plus_one = 1 + exp;
        return 1 + (x - 1) / exp_plus_one - x / exp_plus_one / exp_plus_one;
    });

/**
 * A vector of all the activations.
 */
const std::vector<Activation> activations = {sigmoid, tanh, relu, leaky_relu, linear, swish};

} // namespace Activations

} // namespace FJML
