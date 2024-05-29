// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <cmath>
#include <iostream>

#include "../include/FJML/loss.h"

namespace FJML {

namespace Loss {

float Loss::calc_loss(const Tensor& obs, const Tensor& pred) const { return function(obs, pred); }

Tensor Loss::calc_derivative(const Tensor& obs, const Tensor& pred) const { return derivative(obs, pred); }

/**
 * @brief The mean squared error loss function
 */
const Loss mse(
    "mse",
    [](const Tensor& label, const Tensor& pred) -> float {
        if (label.data_size[0] != pred.data_size[0]) {
            throw std::invalid_argument("The two tensors must have the same size");
        }
        float result = 0;
        for (int i = 0; i < label.data_size[0]; i++) {
            result += (label.data[i] - pred.data[i]) * (label.data[i] - pred.data[i]);
        }
        return result;
    },
    [](const Tensor& label, const Tensor& pred) -> Tensor {
        if (label.data_size[0] != pred.data_size[0]) {
            throw std::invalid_argument("The two tensors must have the same size");
        }
        Tensor result(label.shape, label.device);
        for (int i = 0; i < label.data_size[0]; i++) {
            result.data[i] = 2 * (pred.data[i] - label.data[i]);
        }
        return result;
    });

/**
 * @brief The huber loss function
 */
const Loss huber(
    "huber",
    [](const Tensor& label, const Tensor& pred) -> float {
        if (label.data_size[0] != pred.data_size[0]) {
            throw std::invalid_argument("The two tensors must have the same size");
        }
        float result = 0;
        for (int i = 0; i < label.data_size[0]; i++) {
            float diff = pred.data[i] - label.data[i];
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
    [](const Tensor& label, const Tensor& pred) -> Tensor {
        if (label.data_size[0] != pred.data_size[0]) {
            throw std::invalid_argument("The two tensors must have the same size");
        }
        Tensor result(label.shape, label.device);
        for (int i = 0; i < label.data_size[0]; i++) {
            float diff = pred.data[i] - label.data[i];
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

Loss binary_crossentropy(bool from_logits) {
    if (!from_logits) {
        return Loss(
            "binary_crossentropy",
            [](const Tensor& label, const Tensor& pred) -> float {
                if (label.data_size[0] != pred.data_size[0]) {
                    throw std::invalid_argument("The two tensors must have the same size");
                }
                float result = 0;
                for (int i = 0; i < label.data_size[0]; i++) {
                    result += -label.data[i] * std::log(pred.data[i]) -
                              (1 - label.data[i]) * std::log(1 - pred.data[i]);
                }
                return result;
            },
            [](const Tensor& label, const Tensor& pred) -> Tensor {
                if (label.data_size[0] != pred.data_size[0]) {
                    throw std::invalid_argument("The two tensors must have the same size");
                }
                Tensor result(label.shape, label.device);
                for (int i = 0; i < label.data_size[0]; i++) {
                    result.data[i] = -label.data[i] / pred.data[i] + (1 - label.data[i]) / (1 - pred.data[i]);
                }
                return result;
            });
    }
    return Loss(
        "binary_crossentropy",
        [](const Tensor& label, const Tensor& pred) -> float {
            float result = 0;
            for (int i = 0; i < label.data_size[0]; i++) {
                result += -label.data[i] * pred.data[i] + std::log(1 + std::exp(pred.data[i]));
            }
            return result;
        },
        [](const Tensor& label, const Tensor& pred) -> Tensor {
            Tensor result(label.shape, label.device);
            for (int i = 0; i < label.data_size[0]; i++) {
                float exp_b = std::exp(pred.data[i]);
                result.data[i] = -label.data[i] + exp_b / (1 + exp_b);
            }
            return result;
        });
}

Loss crossentropy(bool from_logits) {
    if (!from_logits) {
        return Loss(
            "crossentropy",
            [](const Tensor& label, const Tensor& pred) -> float {
                if (label.data_size[0] != pred.data_size[0]) {
                    throw std::invalid_argument("The two tensors must have the same size");
                }
                float result = 0;
                for (int i = 0; i < label.data_size[0]; i++) {
                    result += -label.data[i] * std::log(pred.data[i]);
                }
                return result;
            },
            [](const Tensor& label, const Tensor& pred) -> Tensor {
                if (label.data_size[0] != pred.data_size[0]) {
                    throw std::invalid_argument("The two tensors must have the same size");
                }
                Tensor result(label.shape, label.device);
                for (int i = 0; i < label.data_size[0]; i++) {
                    result.data[i] = -label.data[i] / pred.data[i];
                }
                return result;
            });
    }
    return Loss(
        "crossentropy",
        [](const Tensor& label, const Tensor& pred) -> float {
            if (label.data_size[0] != pred.data_size[0] || label.shape[0] != pred.shape[0]) {
                throw std::invalid_argument("The two tensors must have the same size");
            }
            float result = 0;
            for (int datapoint = 0; datapoint < label.shape[0]; datapoint++) {
                int offset = datapoint * label.data_size[1];
                float denom = 0, max = pred.data[offset];
                for (int i = 1; i < label.data_size[1]; i++) {
                    if (pred.data[offset + i] > max) {
                        max = pred.data[offset + i];
                    }
                }
                for (int i = 0; i < label.data_size[1]; i++) {
                    denom += std::exp(pred.data[offset + i] - max);
                }
                for (int i = 0; i < label.data_size[1]; i++) {
                    result += -label.data[offset + i] * (pred.data[offset + i] - max) +
                              label.data[offset + i] * std::log(denom);
                }
            }
            return result;
        },
        [](const Tensor& label, const Tensor& pred) -> Tensor {
            if (label.data_size[0] != pred.data_size[0] || label.shape[0] != pred.shape[0]) {
                throw std::invalid_argument("The two tensors must have the same size");
            }
            Tensor result(label.shape, label.device);
            for (int datapoint = 0; datapoint < label.shape[0]; datapoint++) {
                int offset = datapoint * label.data_size[1];
                float denom = 0, max = pred.data[offset];
                for (int i = 1; i < label.data_size[1]; i++) {
                    if (pred.data[offset + i] > max) {
                        max = pred.data[offset + i];
                    }
                }
                for (int i = 0; i < label.data_size[1]; i++) {
                    denom += std::exp(pred.data[offset + i] - max);
                }
                for (int i = 0; i < label.data_size[1]; i++) {
                    result.data[offset + i] = -label.data[offset + i] + std::exp(pred.data[offset + i] - max) / denom;
                }
            }
            return result;
        });
}

Loss sparse_categorical_crossentropy(bool from_logits) {
    if (!from_logits) {
        return Loss(
            "sparse_categorical_crossentropy",
            [](const Tensor& label, const Tensor& pred) -> float {
                if (label.shape[0] != pred.shape[0]) {
                    throw std::invalid_argument("The two tensors must have the same number of samples");
                }
                float result = 0;
                for (int i = 0; i < label.shape[0]; i++) {
                    result += -std::log(pred.data[static_cast<int>(label.data[i])]);
                }
                return result;
            },
            [](const Tensor& label, const Tensor& pred) -> Tensor {
                if (label.shape[0] != pred.shape[0]) {
                    throw std::invalid_argument("The two tensors must have the same number of samples");
                }
                Tensor result(pred.shape, pred.device);
                for (int i = 0; i < label.shape[0]; i++) {
                    int ind = static_cast<int>(label.data[i]), offset = i * pred.data_size[1];
                    result.data[offset + ind] = -1 / pred.data[offset + ind];
                }
                return result;
            });
    }
    return Loss(
        "sparse_categorical_crossentropy",
        [](const Tensor& label, const Tensor& pred) -> float {
            if (label.shape[0] != pred.shape[0]) {
                throw std::invalid_argument("The two tensors must have the same number of samples");
            }
            float result = 0;
            for (int i = 0; i < label.shape[0]; i++) {
                int offset = i * pred.data_size[1];
                float denom = 0, max = pred.data[offset];
                for (int i = 1; i < label.data_size[1]; i++) {
                    if (pred.data[offset + i] > max) {
                        max = pred.data[offset + i];
                    }
                }
                for (int i = 0; i < pred.data_size[1]; i++) {
                    denom += std::exp(pred.data[offset + i] - max);
                }
                result += -(pred.data[offset + static_cast<int>(label.data[i])] - max) + std::log(denom);
            }
            return result;
        },
        [](const Tensor& label, const Tensor& pred) -> Tensor {
            if (label.shape[0] != pred.shape[0]) {
                throw std::invalid_argument("The two tensors must have the same number of samples");
            }
            Tensor result(pred.shape, pred.device);
            for (int i = 0; i < label.shape[0]; i++) {
                int ind = static_cast<int>(label.data[i]), offset = i * pred.data_size[1];
                float denom = 0, max = pred.data[offset];
                for (int i = 1; i < label.data_size[1]; i++) {
                    if (pred.data[offset + i] > max) {
                        max = pred.data[offset + i];
                    }
                }
                for (int i = 0; i < pred.data_size[1]; i++) {
                    denom += std::exp(pred.data[offset + i] - max);
                }
                for (int i = 0; i < pred.data_size[1]; i++) {
                    result.data[offset + i] = std::exp(pred.data[offset + i] - max) / denom;
                }
                result.data[offset + ind] -= 1;
            }
            return result;
        });
}

} // namespace Loss

} // namespace FJML
