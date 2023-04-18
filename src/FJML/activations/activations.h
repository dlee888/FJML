// Copyright (c) 2022 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef ACTIVATIONS_INCLUDED
#define ACTIVATIONS_INCLUDED

#include <cmath>
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
     * Default constructor
     */
    Activation() : name{"Activation"} {}
    /**
     * Constructor with given name
     */
    Activation(std::string name) : name{name} {}
    /**
     * Virtual destructor
     */
    virtual ~Activation() {}

    /**
     * @brief apply the function to a layer
     * @param layer The layer to apply the function to
     */
    template <typename T> Tensor<T> apply(const Tensor<T>& layer) const {
        Tensor<T> result = Tensor<T>(layer.shape);
        return result;
    }
};

} // namespace Activations

} // namespace FJML

#endif
