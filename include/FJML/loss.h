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
const float clip = 1e6;

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
    std::function<float(const Tensor&, const Tensor&)> function;
    /**
     * @brief The derivative of the loss function
     */
    std::function<Tensor(const Tensor&, const Tensor&)> derivative;

    /**
     * @brief Default constructor
     */
    Loss() = default;

    /**
     * @brief Constructor for the Loss class
     *
     * Each function should take two arguments, the first is the label and the second is the prediction.
     *
     * @param name The name of the loss function
     * @param function The loss function
     * @param derivative The derivative of the loss function
     */
    Loss(std::string name, std::function<float(const Tensor&, const Tensor&)> function,
         std::function<Tensor(const Tensor&, const Tensor&)> derivative)
        : name{name}, function{function}, derivative{derivative} {}

    /**
     * @brief Calculates the loss
     *
     * Returns the sum of the loss function applied to each element of the tensors.
     *
     * @param label The label
     * @param pred The predicted value (function output)
     * @return The loss
     */
    float calc_loss(const Tensor& label, const Tensor& pred) const;

    /**
     * @brief Calculates the derivative of the loss
     *
     * Returns the sum of the derivative of the loss function applied to each element of the tensors.
     *
     * @param label The label
     * @param pred The predicted value (function output)
     * @return The derivative of the loss
     */
    Tensor calc_derivative(const Tensor& label, const Tensor& pred) const;
};

extern const Loss mse, huber;

/**
 * @brief The binary cross entropy loss function
 * @param from_logits Whether the input is from logits (i.e. not sigmoided)
 * @return The binary cross entropy loss function
 */
Loss binary_crossentropy(bool from_logits = false);

/**
 * @brief The cross entropy loss function
 * @param from_logits Whether the input is from logits (i.e. not softmaxed)
 * @return The cross entropy loss function
 */
Loss crossentropy(bool from_logits = false);

/**
 * @brief The sparse categorical cross entropy loss function
 *
 * The label is expected to be a single integer representing the class index.
 */
Loss sparse_categorical_crossentropy(bool from_logits = false);

} // namespace Loss

} // namespace FJML

#endif
