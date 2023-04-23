// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef ACTIVATIONS_INCLUDED
#define ACTIVATIONS_INCLUDED

#include <functional>

#include "../linalg/tensor.h"

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
     * The name of this activation function
     */
    std::string name;
    /**
     * The function to apply to a layer
     */
    std::function<double(double)> func;
    /**
     * The derivative of the function
     */
    std::function<double(double)> derivative;

    /**
     * Constructor with given name and functions
     * @param name The name of the activation function
     * @param func The function to apply to a layer
     * @param derivative The derivative of the function
     */
    Activation(std::string name, std::function<double(double)> func, std::function<double(double)> derivative)
        : name{name}, func{func}, derivative{derivative} {}

    /**
     * @brief apply the function to a layer
     *
     * Note: This function modifies the layer in place.
     *
     * @param layer The layer to apply the function to
     * @return The result of applying the function to the layer
     */
    template <typename T> Tensor<T> apply(Tensor<T>& layer) const {
        for (auto& x : layer) {
            x = func(x);
        }
        return layer;
    }

    /**
     * @brief apply the derivative of the function to a layer
     *
     * Note: This function modifies the layer in place.
     *
     * @param layer The layer to apply the derivative to
     * @return The result of applying the derivative to the layer
     */
    template <typename T> Tensor<T> apply_derivative(Tensor<T>& layer) const {
        for (auto& x : layer) {
            x = derivative(x);
        }
        return layer;
    }

    /**
     * @brief apply the function to a layer
     *
     * Note: This function does not modify the layer.
     *
     * @param layer The layer to apply the function to
     * @return The result of applying the function to the layer
     */
    template <typename T> Tensor<T> forward(const Tensor<T>& layer) const {
        Tensor<T> result = layer;
        for (auto& x : result) {
            x = func(x);
        }
        return result;
    }

    /**
     * @brief apply the derivative of the function to a layer
     *
     * Note: This function does not modify the layer.
     *
     * @param layer The layer to apply the derivative to
     * @return The result of applying the derivative to the layer
     */
    template <typename T> Tensor<T> backward(const Tensor<T>& layer) const {
        Tensor<T> result = layer;
        for (auto& x : result) {
            x = derivative(x);
        }
        return result;
    }
};

extern const Activation sigmoid, tanh, relu, leaky_relu, linear, swish;

} // namespace Activations

} // namespace FJML

#endif
