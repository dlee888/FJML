// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef MLP_INCLUDED
#define MLP_INCLUDED

#include <vector>

#include "layers.h"
#include "linalg.h"
#include "loss.h"
#include "optimizers.h"
#include "tensor.h"

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
     * @brief Default constructor for MLP
     */
    MLP() {}

    /**
     * @brief Constructor for MLP
     * @param layers A vector of layers
     * @param loss The loss function to use
     * @param optimizer The optimizer to use
     */
    MLP(const std::vector<Layers::Layer*>& layers, const Loss::Loss& loss,
        const Optimizers::Optimizer* optimizer = new Optimizers::SGD())
        : layers{layers}, loss_fn{loss} {
        set_optimizer(optimizer);
    }

    /**
     * Destructor for MLP
     */
    ~MLP() {
        for (Layers::Layer* l : layers) {
            delete l;
        }
    }

    /**
     * @brief Set the loss function for the model
     * @param loss The loss function to use
     */
    void set_loss(const Loss::Loss& loss) { loss_fn = loss; }

    /**
     * @brief Set the optimizer for the model
     * @param optimizer The optimizer to use
     */
    void set_optimizer(const Optimizers::Optimizer* optimizer) {
        for (Layers::Layer* l : layers) {
            if (l->name == "Dense") {
                ((Layers::Dense*)l)->set_optimizer(optimizer);
            }
        }
    }

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
    Tensor run(const Tensor& input) const;

    /**
     * @brief Calculate the loss of the model on a batch of data
     * @param x_test The input data
     * @param y_test The target data
     * @return The loss
     */
    double calc_loss(const Tensor& x_test, const Tensor& y_test) const;

    /**
     * @brief Calculate the accuracy of the model on a batch of data
     *
     * Note: data must be in one-hot format
     *
     * @param x_test The input data
     * @param y_test The target data
     * @return The accuracy
     */
    double calc_accuracy(const Tensor& x_test, const Tensor& y_test) const;

    /**
     * Applies gradients in a backwards pass
     *
     * @param input the input
     * @param grads the gradients of the output
     */
    void backwards_pass(const Tensor& input, const Tensor& grads);

    /**
     * @brief Train the model on a batch of data
     *
     * Calculates the loss and gradients for each layer, then applies the optimizer once to update the weights.
     *
     * @param x_train The input data
     * @param y_train The target data
     */
    void grad_descent(const Tensor& x_train, const Tensor& y_train);

    /**
     * @brief Save the model to a file
     * @param filename The name of the file to save to
     */
    void save(std::string filename) const;

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
     */
    void train(const Tensor& x_train, const Tensor& y_train, const Tensor& x_test, const Tensor& y_test, int epochs,
               int batch_size, const std::string& save_file);

    /**
     * @brief Print a summary of the model
     */
    void summary();
};

} // namespace FJML

#endif
