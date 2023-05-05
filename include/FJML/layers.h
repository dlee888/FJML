// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef LAYER_INCLUDED
#define LAYER_INCLUDED

#include <fstream>
#include <string>

#include "activations.h"
#include "linalg.h"
#include "optimizers.h"
#include "tensor.h"

namespace FJML {

/**
 * @brief A namespace for all layers
 *
 * A layer is a function that takes an input and returns an output. A neural network is a composition of layers.
 */
namespace Layers {

/**
 * @brief Based class for all layers
 *
 * An apply function is required for all layers, which takes an input and returns the output.
 * An backward function is also required, equivalent to a backward pass.
 *
 * A save function is also required, which saves the layer's parameters to a file.
 */
class Layer {
  public:
    /**
     * @brief The name of the layer
     */
    std::string name;

    /**
     * @brief Default constructor
     */
    Layer() : name{"Layer"} {}
    /**
     * @brief Constructor
     * @param name The name of the layer
     */
    Layer(std::string name) : name{name} {}
    /**
     * @brief Virtual destructor
     */
    virtual ~Layer() {}

    /**
     * @brief Apply the layer to an input
     * @param input The input to apply the layer to
     * @return The output of the layer
     */
    virtual Tensor apply(const Tensor& input) const { return input; }

    /**
     * @brief Apply the layer to a batch of inputs
     * @param input The batch of inputs to apply the layer to
     * @return The batch of outputs of the layer
     */
    virtual std::vector<Tensor> apply(const std::vector<Tensor>& input) const { return input; }

    /**
     * @brief Backpropagate through the layer
     *
     * Applies gradients to the parameters of the layer.
     *
     * @param input_vals The batch of inputs to apply the layer to
     * @param output_vals The batch of outputs of the layer
     * @param output_grad The batch of gradients of the loss with respect to the output of the layers
     * @return The batch of gradients of the loss with respect to the input of the layer
     */
    virtual std::vector<Tensor> backward(const std::vector<Tensor>& input_vals,
                                         const std::vector<Tensor>& output_grad) {
        return output_grad;
    }

    /**
     * Save the layer to a file
     * @param file The file to save the layer to
     */
    virtual void save(std::ofstream& file) const {}

    /**
     * @brief Print a summary of the layer
     */
    virtual void summary() const {}
};

/**
 * @brief A fully connected layer
 *
 * A fully connected layer is a layer where each node is connected to every node in the previous layer.
 */
class Dense : public Layer {
  public:
    /**
     * @brief The number of nodes in the previous layer
     */
    int input_size;
    /**
     * @brief The number of nodes in this layer
     */
    int output_size;
    /**
     * @brief The weights of the layer, a matrix of shape (input_size, output_size)
     */
    Tensor weights;
    /**
     * @brief The bias of the layer, a vector of shape (output_size)
     */
    Tensor bias;
    /**
     * @brief The activation function of the layer
     */
    Activations::Activation activ;
    /**
     * @brief The optimizer for the weights of the layer
     */
    Optimizers::Optimizer* w_opt;
    /**
     * @brief The optimizer for the bias of the layer
     */
    Optimizers::Optimizer* b_opt;

    /**
     * @brief Constructor for a fully connected layer
     * @param input The number of nodes in the previous layer
     * @param output The number of nodes in this layer
     * @param activ The activation function to use
     * @param opt The optimizer to use for the weights and bias
     * @param randomize Whether to randomize the weights and bias
     */
    Dense(int input, int output, Activations::Activation activ = Activations::sigmoid,
          Optimizers::Optimizer* opt = new Optimizers::SGD(), bool randomize = true);

    /**
     * @brief Load a fully connected layer from a file
     * @param file The file to load the layer from
     */
    Dense(std::ifstream& file);
    /**
     * @brief Destructor
     */
    ~Dense() {}

    /**
     * @brief Apply the layer to an input
     * @param input The input to apply the layer to
     * @return The output of the layer
     */
    Tensor apply(const Tensor& input) const override;

    /**
     * Apply the layer to a batch of inputs
     * @param input The batch of inputs to apply the layer to
     * @return The batch of outputs of the layer
     */
    std::vector<Tensor> apply(const std::vector<Tensor>& input) const override;

    /**
     * @brief Apply the gradient of the layer to a batch of inputs
     * @param input_vals The batch of inputs to apply the layer to
     * @param output_vals The batch of outputs of the layer
     * @param output_grad The batch of gradients of the loss with respect to the output of the layer
     * @return The batch of gradients of the loss with respect to the input of the layer
     */
    std::vector<Tensor> backward(const std::vector<Tensor>& input_vals,
                                 const std::vector<Tensor>& output_grad) override;

    /**
     * @brief Save the layer to a file
     * @param file The file to save the layer to
     */
    void save(std::ofstream& file) const override;

    /**
     * @brief Print a summary of the layer
     */
    void summary() const override;

    /**
     * Set the optimizer for the layer
     * @param opt The optimizer to use for the weights and bias
     */
    void set_optimizer(const Optimizers::Optimizer* opt);
};

/**
 * @brief A softmax layer
 *
 * A softmax layer converts a vector of values into a probability distribution.
 *
 * The softmax function is defined as:
 * \f[\sigma(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}\f]
 */
class Softmax : public Layer {
    Tensor norm(const Tensor& input) const;

  public:
    /**
     * @brief Constructor for a softmax layer
     */
    Softmax() { name = "Softmax"; }
    ~Softmax() {}

    /**
     * @brief Apply the layer to an input
     *
     * The softmax layer applies the softmax function to the input.
     * @param input The input to apply the layer to
     * @return The output of the layer
     */
    Tensor apply(const Tensor& input) const override;

    /**
     * @brief Apply the layer to a batch of inputs
     *
     * The softmax layer applies the softmax function to each input in the batch.
     * @param input The batch of inputs to apply the layer to
     * @return The batch of outputs of the layer
     */
    std::vector<Tensor> apply(const std::vector<Tensor>& input) const override;

    /**
     * @brief Apply the gradient of the layer to a batch of inputs
     * @param input_vals The batch of inputs to apply the layer to
     * @param output_vals The batch of outputs of the layer
     * @param output_grad The batch of gradients of the loss with respect to the output of the layer
     * @return The batch of gradients of the loss with respect to the input of the layer
     */
    std::vector<Tensor> backward(const std::vector<Tensor>& input_vals,
                                 const std::vector<Tensor>& output_grad) override;

    /**
     * @brief Save the layer to a file
     * @param file The file to save the layer to
     */
    void save(std::ofstream& file) const override;

    /**
     * @brief Print a summary of the layer
     */
    void summary() const override;
};

/**
 * @brief Load a layer from a file
 * @param file The file to load the layer from
 * @return A pointer to the loaded layer
 */
Layer* load(std::ifstream& file);

} // namespace Layers

} // namespace FJML

#endif
