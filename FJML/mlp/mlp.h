#ifndef MLP_INCLUDED
#define MLP_INCLUDED

#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "../layers/layers.h"
#include "../loss/loss.h"
#include "../optimizers/optimizers.h"
#include "../linalg/linalg.h"
#include "../linalg/tensor.h"

namespace FJML {

/**
 * @brief Multi-layer perceptron class
 *
 * This class implements a multi-layer perceptron with a variable number of layers.
 */
class MLP {
  public:
	/**
	 * @brief This is a vector containing pointers to the layers of the MLP.
	 */
	std::vector<Layers::Layer*> layers;
	/**
	 * @brief This is the loss function used by the MLP.
	 */
	Loss::Loss loss_fn;

	/**
	 * @brief No args constructor
	 *
	 * This constructor creates an empty MLP with no set loss function.
	 */
	MLP() {}

	/**
	 * @brief Constructor for MLP
	 * @param _layers A vector of layers
	 * @param _loss The loss function to use
	 * @param _optimizer The optimizer to use
	 */
	MLP(std::vector<Layers::Layer*> _layers, Loss::Loss _loss,
		Optimizers::Optimizer<1>* _optimizer = new Optimizers::SGD<1>)
		: layers{_layers}, loss_fn{_loss} {
		set_optimizer(_optimizer);
	}

	~MLP() {
		for (Layers::Layer* l : layers) {
			delete l;
		}
	}

	/**
	 * @brief Set the loss function for the model
	 * @param loss The loss function to use
	 */
	void set_loss(Loss::Loss loss) { loss_fn = loss; }

	/**
	 * @brief Set the optimizer for the model
	 * @param optimizer The optimizer to use
	 */
	void set_optimizer(Optimizers::Optimizer<1>* optimizer) {
		for (Layers::Layer* l : layers) {
			if (l->name == "Dense") {
				((Layers::Dense*)l)->w_opt = Optimizers::get_optimizer<2>(optimizer);
				((Layers::Dense*)l)->b_opt = Optimizers::get_optimizer<1>(optimizer);
			}
		}
	}

	/**
	 * @brief Add a layer to the model
	 * @param layer The layer to add
	 */
	void add(Layers::Layer layer) { layers.push_back(&layer); }

	/**
	 * @brief Add a layer to the model
	 * @param layer The layer to add
	 */
	void add(Layers::Layer* layer) { layers.push_back(layer); }

	/**
	 * @brief Run the model on an input
	 *
	 * This applies all the layers in the model to the input.
	 *
	 * @param input The input to run the model on
	 */
	layer_vals run(layer_vals input);

	/**
	 * @brief Calculate the loss of the model on a batch of data
	 * @param x_test The input data
	 * @param y_test The target data
	 * @return The loss
	 */
	double calc_loss(std::vector<layer_vals> x_test, std::vector<layer_vals> y_test);

	/**
	 * @brief Calculate the accuracy of the model on a batch of data
	 *
	 * Note: data must be in one-hot format
	 *
	 * @param x_test The input data
	 * @param y_test The target data
	 * @return The accuracy
	 */
	double calc_accuracy(std::vector<layer_vals> x_test, std::vector<layer_vals> y_test);

	/**
	 * @brief Train the model on a batch of data
	 *
	 * Calculates the loss and gradients for each layer, then applies the optimizer once to update the weights.
	 *
	 * @param x_train The input data
	 * @param y_train The target data
	 * @param mask A mask to apply to the data
	 */
	void grad_descent(std::vector<layer_vals> x_train, std::vector<layer_vals> y_train,
					  std::vector<std::vector<bool>>* mask = nullptr);

	/**
	 * @brief Save the model to a file
	 * @param filename The name of the file to save to
	 */
	void save(std::string filename);

	/**
	 * @brief Load the model from a file
	 * @param filename The name of the file to load from
	 */
	void load(std::string filename);

	/**
	 * @brief Train the model on a batch of data
	 * @param x_train The input data
	 * @param y_train The target data
	 * @param x_test The input data to test on
	 * @param y_test The target data to test on
	 * @param epochs The number of epochs to train for
	 * @param batch_size The size of the batches to train on
	 * @param save_file The file to save the model to, or "" to not save
	 * @param mask A mask to apply to the data
	 */
	void train(std::vector<layer_vals> x_train, std::vector<layer_vals> y_train, std::vector<layer_vals> x_test,
			   std::vector<layer_vals> y_test, int epochs, int batch_size, std::string save_file,
			   std::vector<std::vector<bool>>* mask = nullptr);

	/**
	 * @brief Print a summary of the model
	 */
	void summary();
};

} // namespace FJML

#endif
