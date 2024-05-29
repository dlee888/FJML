// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef OPTIMIZERS_INCLUDED
#define OPTIMIZERS_INCLUDED

#include <string>

#include "linalg.h"
#include "tensor.h"

namespace FJML {

/**
 * @brief Namespace for optimizers
 *
 * Optimizers are used to optimize the parameters of a neural network during gradient descent.
 */
namespace Optimizers {

/**
 * @brief Based class for all optimizers
 *
 * Optimizes an tensor during gradient descent
 *
 * Each optimizer must implement the apply_grad method, which takes parameters and gradients, and updates the
 * parameters.
 */
class Optimizer {
  public:
    /**
     * @brief The name of the optimizer
     */
    std::string name;

    /**
     * @brief Default constructor
     */
    Optimizer() : name{"Optimizer"} {}
    /**
     * @brief Constructor
     * @param name The name of the optimizer
     */
    Optimizer(std::string name) : name{name} {}
    /**
     * @brief Virtual destructor
     */
    virtual ~Optimizer() {}

    /**
     * @brief Applies the gradient to the parameters
     * @param params The parameters to be updated
     * @param grads The gradients to be applied
     */
    virtual void apply_grad(Tensor& params, const Tensor& grads) {}

    /**
     * Clone the optimizer
     * @return A pointer to a copy of the optimizer
     */
    virtual Optimizer* clone() const { return new Optimizer(*this); }
};

/**
 * @brief Stochastic Gradient Descent
 * @details Optimizes an N dimensional tensor during gradient descent. The optimizer updates the parameters by
 * subtracting the learning rate times the gradient from the parameters.
 */
class SGD : public Optimizer {
  public:
    /**
     * @brief The learning rate
     */
    float alpha;

    /**
     * @brief Default constructor
     */
    SGD(float learning_rate = 0.01) : Optimizer{"SGD"}, alpha{learning_rate} {}
    /**
     * @brief Destructor
     */
    ~SGD() {}

    /**
     * @brief Applies the gradient to the parameters
     * @param params The parameters to be updated
     * @param grads The gradients to be applied
     */
    void apply_grad(Tensor& params, const Tensor& grads) override;

    /**
     * @brief Clone the optimizer
     *
     * Note: this should clone the hyperparameters but not the state
     *
     * @return A pointer to a copy of the optimizer
     */
    Optimizer* clone() const override { return new SGD(this->alpha); }
};

/**
 * @brief Adam optimizer
 * @details Optimizes an N dimensional tensor during gradient descent using the Adam algorithm.
 */
class Adam : public Optimizer {
    /**
     * @brief The first momentum
     */
    Tensor m;
    /**
     * @brief The second momentum
     */
    Tensor v;
    /**
     * @brief The time step
     */
    int t = 1;

    /**
     * @brief Helper function to initialize the first and second momentums
     * @param params The parameters to be updated
     */
    void init(const Tensor& params);

  public:
    /**
     * @brief The epsilon value
     */
    static constexpr float epsilon = 1e-8;

    /**
     * @brief The learning rate
     */
    float alpha;
    /**
     * @brief The first momentum
     */
    float beta1;
    /**
     * @brief The second momentum
     */
    float beta2;

    /**
     * @brief Constructor for Adam
     * @param a The learning rate
     * @param b1 The first momentum
     * @param b2 The second momentum
     */
    Adam(float a = 0.001, float b1 = 0.9, float b2 = 0.999)
        : Optimizer{"Adam"}, t{1}, alpha{a}, beta1{b1}, beta2{b2} {}
    /**
     * @brief Destructor
     */
    ~Adam() {}

    /**
     * @brief Applies the gradient to the parameters
     *
     * @param params The parameters to be updated
     * @param grads The gradients to be applied
     */
    void apply_grad(Tensor& params, const Tensor& grads) override;

    /**
     * @brief Clone the optimizer
     * @return A pointer to a copy of the optimizer
     */
    Optimizer* clone() const override { return new Adam(this->alpha, this->beta1, this->beta2); }
};

} // namespace Optimizers

} // namespace FJML

#endif
