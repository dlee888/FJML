// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <cmath>

#include "../include/FJML/loss.h"

namespace FJML {

namespace Loss {

double Loss::calc_loss(const Tensor& obs, const Tensor& pred) const { return function(obs, pred); }

Tensor Loss::calc_derivative(const Tensor& obs, const Tensor& pred) const { return derivative(obs, pred); }

/**
 * @brief The mean squared error loss function
 */
const Loss mse(
    "mse",
    [](const Tensor& a, const Tensor& b) -> double {
        double result = 0;
        for (int i = 0; i < a.data_size[0]; i++) {
            result += (a.data[i] - b.data[i]) * (a.data[i] - b.data[i]);
        }
        return result;
    },
    [](const Tensor& a, const Tensor& b) {
        Tensor result(a.shape, a.device);
        for (int i = 0; i < a.data_size[0]; i++) {
            result.data[i] = 2 * (b.data[i] - a.data[i]);
        }
        return result;
    });
/**
 * @brief The huber loss function
 */
const Loss huber(
    "huber",
    [](const Tensor& a, const Tensor& b) -> double {
        double result = 0;
        for (int i = 0; i < a.data_size[0]; i++) {
            double diff = b.data[i] - a.data[i];
            if (diff < -1) {
                result += -2 * diff - 1;
            } else if (diff > 1) {
                result += 2 * diff - 1;
            } else {
                result += diff * diff;
            }
        }
        return result;
    },
    [](const Tensor& a, const Tensor& b) -> Tensor {
        Tensor result(a.shape, a.device);
        for (int i = 0; i < a.data_size[0]; i++) {
            double diff = b.data[i] - a.data[i];
            if (diff < -1) {
                result.data[i] = -2;
            } else if (diff > 1) {
                result.data[i] = 2;
            } else {
                result.data[i] = 2 * diff;
            }
        }
        return result;
    });
/**
 * @brief The cross entropy loss function
 * @param from_logits Whether the input is from logits (i.e. not softmaxed)
 * @return The cross entropy loss function
 */
Loss crossentropy(bool from_logits) {
    if (!from_logits) {
        return Loss(
            "crossentropy",
            [](const Tensor& a, const Tensor& b) -> double {
                double result = 0;
                for (int i = 0; i < a.data_size[0]; i++) {
                    result += a.data[i] * std::log(b.data[i]) + (1 - a.data[i]) * std::log(1 - b.data[i]);
                }
                return -result;
            },
            [](const Tensor& a, const Tensor& b) -> Tensor {
                Tensor result(a.shape, a.device);
                for (int i = 0; i < a.data_size[0]; i++) {
                    result.data[i] = (b.data[i] - a.data[i]) / (b.data[i] * (1 - b.data[i]));
                }
                return result;
            });
    }
    return Loss(
        "crossentropy",
        [](const Tensor& a, const Tensor& b) -> double {
            double result = 0;
            for (int i = 0; i < a.data_size[0]; i++) {
                result += a.data[i] * b.data[i];
            }
            return -result;
        },
        [](const Tensor& a, const Tensor& b) -> Tensor {
            Tensor result(a.shape, a.device);
            for (int i = 0; i < a.data_size[0]; i++) {
                result.data[i] = -a.data[i];
            }
            return result;
        });
}

/**
 * @brief The sparse categorical cross entropy loss function
 *
 * The label is expected to be a single integer representing the class index.
 */
Loss sparse_categorical_crossentropy(bool from_logits) {
    if (!from_logits) {
        return Loss(
            "sparse_categorical_crossentropy",
            [](const Tensor& a, const Tensor& b) -> double { return -std::log(b.data[static_cast<int>(a.data[0])]); },
            [](const Tensor& a, const Tensor& b) -> Tensor {
                Tensor result(b.shape, b.device);
                result.data[static_cast<int>(a.data[0])] = -1 / b.data[static_cast<int>(a.data[0])];
                return result;
            });
    }
    return Loss(
        "sparse_categorical_crossentropy",
        [](const Tensor& a, const Tensor& b) -> double { return -b.data[static_cast<int>(a.data[0])]; },
        [](const Tensor& a, const Tensor& b) -> Tensor {
            Tensor result(b.shape, b.device);
            result.data[static_cast<int>(a.data[0])] = -1;
            return result;
        });
}

} // namespace Loss

} // namespace FJML
