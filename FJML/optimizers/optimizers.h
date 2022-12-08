#ifndef OPTIMIZERS_INCLUDED
#define OPTIMIZERS_INCLUDED

#include <cmath>
#include <iostream>

#include "../util/linalg.h"
#include "../util/types.h"

namespace FJML {

namespace Optimizers {

template <int N> class Optimizer {
  public:
	std::string name;

	Optimizer() {}
	Optimizer(std::string _name) : name{_name} {}
	virtual ~Optimizer() {}

	virtual void apply_grad(Tensor<N>& params, const Tensor<N>& grads) {}
};

template <int N> class SGD : public Optimizer<N> {
  public:
	double alpha;

	SGD(double a = 0.01) : Optimizer<N>{"SGD"}, alpha{a} {}
	~SGD() {}

	void apply_grad(Tensor<N>& params, const Tensor<N>& grads) { params -= grads * alpha; }
};

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

template <int N> Optimizer<N>* get_optimizer(Optimizer<1>* x) {
	if (x->name == "SGD") {
		return new SGD<N>(((SGD<1>*)x)->alpha);
	} else if (x->name == "Adam") {
		return new Adam<N>(((Adam<1>*)x)->alpha, ((Adam<1>*)x)->beta1, ((Adam<1>*)x)->beta2);
	}
	std::cout << "No optimizer found for name " << x->name << std::endl;
	return nullptr;
}

} // namespace Optimizers

} // namespace FJML

#endif