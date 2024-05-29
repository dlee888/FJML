// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef ACTIVATIONS_INCLUDED
#define ACTIVATIONS_INCLUDED

#include <functional>

#include "tensor.h"

namespace FJML {

/**
 * @brief Activation functions
 *
 * @details Activation functions are used to transform the output of a neuron.
 */
namespace Activations {

/**
 * @brief This class represents a generic activation function
 *
 * @details This class represents a generic activation function.
 */
class Activation {
  public:
    /**
     * @brief The name of this activation function
     */
    std::string name;
    /**
     * @brief The function to apply to a layer
     */
    std::function<float(float)> func;
    /**
     * @brief The derivative of the function
     */
    std::function<float(float)> derivative;

    /**
     * @brief Constructor with given name and functions
     * @param name The name of the activation function
     * @param func The function to apply to a layer
     * @param derivative The derivative of the function
     */
    Activation(std::string name, std::function<float(float)> func, std::function<float(float)> derivative);

    /**
     * @brief apply the function to a layer
     *
     * Note: This function modifies the layer in place.
     *
     * @param layer The layer to apply the function to
     * @return The result of applying the function to the layer
     */
    Tensor apply(Tensor& layer) const;

    /**
     * @brief apply the derivative of the function to a layer
     *
     * Note: This function modifies the layer in place.
     *
     * @param layer The layer to apply the derivative to
     * @return The result of applying the derivative to the layer
     */
    Tensor apply_derivative(Tensor& layer) const;

    /**
     * @brief apply the function to a layer
     *
     * Note: This function does not modify the layer.
     *
     * @param layer The layer to apply the function to
     * @return The result of applying the function to the layer
     */
    Tensor forward(const Tensor& layer) const;

    /**
     * @brief apply the derivative of the function to a layer
     *
     * Note: This function does not modify the layer.
     *
     * @param layer The layer to apply the derivative to
     * @return The result of applying the derivative to the layer
     */
    Tensor backward(const Tensor& layer) const;
};

extern const Activation sigmoid, tanh, relu, leaky_relu, linear, swish;

extern const std::vector<Activation> activations;

} // namespace Activations

} // namespace FJML

#endif
