// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include "../include/FJML/metrics.h"
#include "../include/FJML/linalg.h"

namespace FJML {

namespace MLP {

Metric::Metric(std::string name, std::function<float(const Tensor&, const Tensor&)> compute)
    : name{name}, compute{compute} {}

/**
 * @brief This is the accuracy metric.
 *
 * The accuracy metric is defined as the percentage of correct predictions.
 */
Metric accuracy{"accuracy", [](const Tensor& label, const Tensor& output) -> float {
                    Tensor label_argmax = LinAlg::argmax(label, 1);
                    Tensor output_argmax = LinAlg::argmax(output, 1);
                    return LinAlg::mean(LinAlg::equal(label_argmax, output_argmax));
                }};

/**
 * @brief This is the mean squared error metric.
 *
 * The mean squared error metric is defined as the mean of the squared difference between the label and the output.
 */
Metric mean_squared_error{"mean_squared_error", [](const Tensor& label, const Tensor& output) -> float {
                              return LinAlg::mean(LinAlg::pow(label - output, 2));
                          }};

/**
 * @brief This is the sparse categorical accuracy metric.
 *
 * The sparse categorical accuracy metric is defined as the percentage of correct predictions.
 *
 * Here each label is an integer representing the class.
 */
Metric sparse_categorical_accuracy{"sparse_categorical_accuracy",
                                   [](const Tensor& label, const Tensor& output) -> float {
                                       Tensor output_argmax = LinAlg::argmax(output, 1);
                                       int correct = 0;
                                       for (int i = 0; i < label.shape[0]; i++) {
                                           if (label.data[i] == output_argmax.data[i]) {
                                               correct++;
                                           }
                                       }
                                       return (float)correct / ((float)label.shape[0]);
                                   }};

} // namespace MLP

} // namespace FJML
