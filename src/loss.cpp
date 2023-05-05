// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <cmath>

#include "../include/FJML/loss.h"

namespace FJML {

namespace Loss {

double Loss::calc_loss(const Tensor& obs, const Tensor& pred) const {
    if (obs.data_size[0] != pred.data_size[0]) {
        throw std::invalid_argument("obs and pred must have the same number of items");
    }
    double loss = 0;
    for (int i = 0; i < obs.data_size[0]; i++) {
        loss += function(obs.data[i], pred.data[i]);
    }
    return loss;
}

Tensor Loss::calc_derivative(const Tensor& obs, const Tensor& pred) const {
    if (obs.data_size[0] != pred.data_size[0]) {
        throw std::invalid_argument("obs and pred must have the same number of items");
    }
    Tensor deriv = Tensor(pred.shape);
    for (int i = 0; i < obs.data_size[0]; i++) {
        deriv.data[i] = std::max(std::min(derivative(obs.data[i], pred.data[i]), clip), -clip);
    }
    return deriv;
}

/**
 * @brief The mean squared error loss function
 */
const Loss mse(
    "mse", [](double a, double b) { return (a - b) * (a - b); }, [](double a, double b) { return 2 * (b - a); });
/**
 * @brief The huber loss function
 */
const Loss huber(
    "huber",
    [](double a, double b) -> double {
        double diff = a - b;
        if (std::abs(diff) > 1) {
            return std::abs(diff) - 0.5;
        }
        return diff * diff / 2;
    },
    [](double a, double b) -> double {
        double diff = a - b;
        if (diff < -1) {
            return 1;
        }
        if (diff > 1) {
            return -1;
        }
        return (b - a);
    });
/**
 * @brief The cross entropy loss function
 */
const Loss crossentropy(
    "crossentropy",
    [](double a, double b) -> double {
        if (a == 0) {
            return -log(1 - b);
        }
        return -log(b);
    },
    [](double a, double b) -> double {
        if (a == 0) {
            return 1 / (1 - b);
        }
        return -1 / b;
    });

} // namespace Loss

} // namespace FJML
