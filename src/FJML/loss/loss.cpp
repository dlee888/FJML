// Copyright (c) 2022 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <cmath>

#include "loss.h"

namespace FJML {

namespace Loss {

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
