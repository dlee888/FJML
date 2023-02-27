#ifndef OPTIMIZERS_INCLUDED
#define OPTIMIZERS_INCLUDED

#include <cmath>
#include <iostream>

#include "../linalg/linalg.h"
#include "../linalg/tensor.h"

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
 * Optimizes an N dimensional tensor during gradient descent
 *
 * Each optimizer must implement the apply_grad method, which takes parameters and gradients, and updates the
 * parameters.
 *
 * @tparam N The number of dimensions of the tensor to be optimized
 */
template <int N> class Optimizer {
  public:
	std::string name;

	Optimizer() {}
	Optimizer(std::string _name) : name{_name} {}
	virtual ~Optimizer() {}

	virtual void apply_grad(Tensor<N>& params, const Tensor<N>& grads) {}
};

/**
 * @brief Stochastic Gradient Descent
 * @details Optimizes an N dimensional tensor during gradient descent. The optimizer updates the parameters by
 * subtracting the learning rate times the gradient from the parameters.
 * @tparam N The number of dimensions of the tensor
 */
template <int N> class SGD : public Optimizer<N> {
  public:
	double alpha;

	SGD(double a = 0.01) : Optimizer<N>{"SGD"}, alpha{a} {}
	~SGD() {}

	void apply_grad(Tensor<N>& params, const Tensor<N>& grads) { params -= grads * alpha; }
};

/**
 * @brief Adam optimizer
 * @details Optimizes an N dimensional tensor during gradient descent using the Adam algorithm.
 * @tparam N The number of dimensions of the tensor
 */
template <int N> class Adam : public Optimizer<N> {
	void init(Tensor<N> params) {
		if (m.size() == 0) {
			m = Tensor<N>(params.shape);
			v = Tensor<N>(params.shape);
		}
	}

  public:
	static constexpr double eps = 1e-8;

	double alpha, beta1, beta2;
	Tensor<N> m, v;

	Adam(double a = 0.001, double b1 = 0.9, double b2 = 0.999) : Optimizer<N>{"Adam"}, alpha{a}, beta1{b1}, beta2{b2} {}
	~Adam() {}

	void apply_grad(Tensor<N>& params, const Tensor<N>& grads) {
		init(params);
		m = beta1 * m + (1 - beta1) * grads;
		v = beta2 * v + (1 - beta2) * grads * grads;
		Tensor<N> m_hat = m / (1 - beta1);
		Tensor<N> v_hat = v / (1 - beta2);
		params -= alpha * m_hat / (v_hat.apply_fn([](double x) { return std::sqrt(x); }) + Tensor<N>(v_hat.shape, eps));
	}
};

/**
 * @brief Creates a new optimizer with the same hyperparameters as the given optimizer
 * @tparam N The number of dimensions of the tensor
 * @param opt The optimizer to copy
 * @return A new optimizer with the same hyperparameters as the given optimizer
 */
template <int N> Optimizer<N>* get_optimizer(Optimizer<1>* opt) {
	if (opt->name == "SGD") {
		return new SGD<N>(((SGD<1>*)opt)->alpha);
	} else if (opt->name == "Adam") {
		return new Adam<N>(((Adam<1>*)opt)->alpha, ((Adam<1>*)opt)->beta1, ((Adam<1>*)opt)->beta2);
	}
	std::cout << "No optimizer found for name " << opt->name << std::endl;
	return nullptr;
}

} // namespace Optimizers

} // namespace FJML

#endif
