// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef OPTIMIZERS_INCLUDED
#define OPTIMIZERS_INCLUDED

#include <iostream>
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
    virtual void apply_grad(Tensor<double>& params, const Tensor<double>& grads) {
        std::cerr << "Optimizer not implemented" << std::endl;
    }

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
    double alpha;

    /**
     * @brief Default constructor
     */
    SGD(double learning_rate = 0.01) : Optimizer{"SGD"}, alpha{learning_rate} {}
    /**
     * @brief Destructor
     */
    ~SGD() {}

    /**
     * @brief Applies the gradient to the parameters
     * @param params The parameters to be updated
     * @param grads The gradients to be applied
     */
    void apply_grad(Tensor<double>& params, const Tensor<double>& grads) override;

    /**
     * Clone the optimizer
     * @return A pointer to a copy of the optimizer
     */
    Optimizer* clone() const override { return new SGD(*this); }
};

/**
 * @brief Adam optimizer
 * @details Optimizes an N dimensional tensor during gradient descent using the Adam algorithm.
 */
class Adam : public Optimizer {
    /**
     * @brief The first momentum
     */
    Tensor<double> m;
    /**
     * @brief The second momentum
     */
    Tensor<double> v;
    /**
     * @brief The time step
     */
    int t = 1;

    /**
     * Helper function to initialize the first and second momentums
     * @param params The parameters to be updated
     */
    void init(const Tensor<double>& params);

  public:
    /**
     * @brief The epsilon value
     */
    static constexpr double epsilon = 1e-8;

    /**
     * @brief The learning rate
     */
    double alpha;
    /**
     * @brief The first momentum
     */
    double beta1;
    /**
     * @brief The second momentum
     */
    double beta2;

    /**
     * Constructor
     * @param a The learning rate
     * @param b1 The first momentum
     * @param b2 The second momentum
     */
    Adam(double a = 0.001, double b1 = 0.9, double b2 = 0.999)
        : Optimizer{"Adam"}, m{{0}}, v{{0}}, t{1}, alpha{a}, beta1{b1}, beta2{b2} {}
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
    void apply_grad(Tensor<double>& params, const Tensor<double>& grads) override;

    /**
     * Clone the optimizer
     * @return A pointer to a copy of the optimizer
     */
    Optimizer* clone() const override { return new Adam(*this); }
};

} // namespace Optimizers

} // namespace FJML

#endif
