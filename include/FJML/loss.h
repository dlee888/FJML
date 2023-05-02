// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef LOSS_INCLUDED
#define LOSS_INCLUDED

#include <functional>
#include <string>

#include "tensor.h"

namespace FJML {

/**
 * @brief Namespace for loss functions
 */
namespace Loss {

/**
 * @brief The clip value for the derivative of the loss function
 */
const double clip = 1e6;

/**
 * @brief Loss function class
 *
 * The loss function class is a functor that takes two arguments, the function and its derivative.
 * The functions should take two arguments, the first is the label and the second is the prediction.
 */
class Loss {
  public:
    /**
     * @brief The name of the loss function
     */
    std::string name;
    /**
     * @brief The loss function
     */
    std::function<double(double, double)> function;
    /**
     * @brief The derivative of the loss function
     */
    std::function<double(double, double)> derivative;

    /**
     * Constructor
     *
     * Each function should take two arguments, the first is the label and the second is the prediction.
     *
     * @param name The name of the loss function
     * @param function The loss function
     * @param derivative The derivative of the loss function
     */
    Loss(std::string name, std::function<double(double, double)> function,
         std::function<double(double, double)> derivative)
        : name{name}, function{function}, derivative{derivative} {}

    /**
     * @brief Calculates the loss
     *
     * Returns the sum of the loss function applied to each element of the tensors.
     *
     * @param obs The observed value (label)
     * @param pred The predicted value (function output)
     * @return The loss
     */
    double calc_loss(const Tensor<double>& obs, const Tensor<double>& pred) const {
        if (obs.data_size[0] != pred.data_size[0]) {
            throw std::invalid_argument("obs and pred must have the same number of items");
        }
        double loss = 0;
        for (int i = 0; i < obs.data_size[0]; i++) {
            loss += function(obs.data[i], pred.data[i]);
        }
        return loss;
    }

    /**
     * @brief Calculates the derivative of the loss
     *
     * Returns the sum of the derivative of the loss function applied to each element of the tensors.
     *
     * @param obs The observed value (label)
     * @param pred The predicted value (function output)
     * @return The derivative of the loss
     */
    Tensor<double> calc_derivative(const Tensor<double>& obs, const Tensor<double>& pred) const {
        if (obs.data_size[0] != pred.data_size[0]) {
            throw std::invalid_argument("obs and pred must have the same number of items");
        }
        Tensor<double> deriv = Tensor<double>(pred.shape);
        for (int i = 0; i < obs.data_size[0]; i++) {
            deriv.data[i] = std::max(std::min(derivative(obs.data[i], pred.data[i]), clip), -clip);
        }
        return deriv;
    }
};

extern const Loss mse, huber, crossentropy;

} // namespace Loss

} // namespace FJML

#endif
