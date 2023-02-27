#ifndef LAYER_INCLUDED
#define LAYER_INCLUDED

#include <cassert>
#include <chrono>
#include <climits>
#include <fstream>
#include <iostream>
#include <random>

#include "../activations/activations.h"
#include "../optimizers/optimizers.h"
#include "../linalg/linalg.h"
#include "../linalg/tensor.h"

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
 * An apply_grad function is also required, equivalent to a backwards pass.
 *
 * A save function is also required, which saves the layer's parameters to a file.
 */
class Layer {
  public:
	std::string name;

	Layer() {}
	Layer(std::string _name) : name{_name} {}
	virtual ~Layer() {}

	virtual layer_vals apply(const layer_vals& input);
	virtual std::vector<layer_vals> apply(const std::vector<layer_vals>& input);

	// Return the gradient w.r.t the input nodes of this layer
	virtual std::vector<layer_vals> apply_grad(const std::vector<layer_vals>& input_vals,
											   const std::vector<layer_vals>& output_grad);

	virtual void save(std::ofstream& file);

	virtual void summary() {}
};

/**
 * @brief A fully connected layer
 *
 * A fully connected layer is a layer where each node is connected to every node in the previous layer.
 */
class Dense : public Layer {
  public:
	int input_size, output_size;
	weights layer_weights;
	bias layer_bias;
	Activations::Activation activ;
	Optimizers::Optimizer<2>* w_opt;
	Optimizers::Optimizer<1>* b_opt;

	Dense();
	/**
	 * @brief Constructor for a fully connected layer
	 * @param _input_size The number of nodes in the previous layer
	 * @param _output_size The number of nodes in this layer
	 * @param _activ The activation function to use
	 * @param _opt The optimizer to use for the weights and bias
	 * @param randomize Whether to randomize the weights and bias
	 */
	Dense(int _input, int _output, Activations::Activation _activ = Activations::sigmoid,
		  Optimizers::Optimizer<1>* _opt = new Optimizers::SGD<1>, bool randomize = true);

	/**
	 * @brief Load a fully connected layer from a file
	 * @param file The file to load the layer from
	 */
	Dense(std::ifstream& file);
	~Dense() {}

	/**
	 * @brief Apply the layer to an input
	 * @param input The input to apply the layer to
	 * @return The output of the layer
	 */
	layer_vals apply(const layer_vals& input);

	/**
	 * @brief Apply the layer to a batch of inputs
	 * @param input The batch of inputs to apply the layer to
	 * @return The batch of outputs of the layer
	 */
	std::vector<layer_vals> apply(const std::vector<layer_vals>& input);

	/**
	 * @brief Apply the gradient of the layer to a batch of inputs
	 * @param input_vals The batch of inputs to apply the layer to
	 * @param output_grad The batch of gradients of the loss with respect to the output of the layer
	 * @return The batch of gradients of the loss with respect to the input of the layer
	 */
	std::vector<layer_vals> apply_grad(const std::vector<layer_vals>& input_vals,
									   const std::vector<layer_vals>& output_grad);

	/**
	 * @brief Save the layer to a file
	 * @param file The file to save the layer to
	 */
	void save(std::ofstream& file);

	/**
	 * @brief Print a summary of the layer
	 */
	void summary();
};

/**
 * @brief A softmax layer
 *
 * A softmax layer converts a vector of values into a probability distribution.
 *
 * The softmax function is defined as:
 * \[ \sigma(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}} \]
 */
class Softmax : public Layer {
	void norm(layer_vals& input);

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
	layer_vals apply(const layer_vals& input);

	/**
	 * @brief Apply the layer to a batch of inputs
	 * @param input The batch of inputs to apply the layer to
	 * @return The batch of outputs of the layer
	 * @see apply(const layer_vals& input)
	 */
	std::vector<layer_vals> apply(const std::vector<layer_vals>& input);

	/**
	 * @brief Apply the gradient of the layer to a batch of inputs
	 * @param input_vals The batch of inputs to apply the layer to
	 * @param output_grad The batch of gradients of the loss with respect to the output of the layer
	 * @return The batch of gradients of the loss with respect to the input of the layer
	 */
	std::vector<layer_vals> apply_grad(const std::vector<layer_vals>& input_vals,
									   const std::vector<layer_vals>& output_grad);

	/**
	 * @brief Save the layer to a file
	 * @param file The file to save the layer to
	 */
	void save(std::ofstream& file);

	/**
	 * @brief Print a summary of the layer
	 */
	void summary();
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
