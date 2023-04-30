// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <cmath>

#include "../../../include/FJML/activations.h"

namespace FJML {

namespace Activations {

/**
 * The sigmoid function.
 *
 * Equation:
 * \f[
 *   \sigma(x) = \frac{1}{1 + e^{-x}}
 * \f]
 */
const Activation sigmoid = Activation(
    "sigmoid", [](double x) { return 1 / (1 + std::exp(-x)); },
    [](double x) { return std::exp(-x) / std::pow(1 + std::exp(-x), 2); });

/**
 * The hyperbolic tangent function.
 *
 * Equation:
 * \f[
 *  \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
 * \f]
 */
const Activation tanh = Activation(
    "tanh", [](double x) { return std::tanh(x); }, [](double x) { return 1 - std::pow(std::tanh(x), 2); });

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
    "relu", [](double x) { return x > 0 ? x : 0; }, [](double x) { return x > 0 ? 1 : 0; });

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
    "leaky relu", [](double x) { return x > 0 ? x : 0.01 * x; }, [](double x) { return x > 0 ? 1 : 0.01; });

/**
 * The linear function.
 *
 * Equation:
 * \f[
 * \text{linear}(x) = x
 * \f]
 */
const Activation linear = Activation(
    "linear", [](double x) { return x; }, [](double x) { return 1; });

/**
 * The swish function.
 *
 * Equation:
 * \f[
 * \text{swish}(x) = \frac{x}{1 + e^{-x}}
 * \f]
 */
const Activation swish = Activation(
    "swish", [](double x) { return x / (1 + std::exp(-x)); },
    [](double x) {
        double exp = std::exp(x);
        double exp_plus_one = 1 + exp;
        return 1 + (x - 1) / exp_plus_one - x / exp_plus_one / exp_plus_one;
    });

/**
 * A vector of all the activations.
 */
const std::vector<Activation> activations = {sigmoid, tanh, relu, leaky_relu, linear, swish};

} // namespace Activations

} // namespace FJML
