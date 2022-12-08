#ifndef ACTIVATIONS_INCLUDED
#define ACTIVATIONS_INCLUDED

#include <cmath>
#include <functional>

#include "../util/types.h"

namespace FJML {

namespace Activations {

class Activation {
  public:
	std::function<double(double)> func, grad;
	std::string name;

	Activation() {}
	Activation(std::function<double(double)> f, std::function<double(double)> g, std::string n) {
		func = f;
		grad = g;
		name = n;
	}

	void apply(layer_vals& l) {
		for (auto& i : l) {
			i = func(i);
		}
	}
};

extern Activation linear, sigmoid, swish, relu, leaky_relu, tanh;

extern std::vector<Activation> all_activations;

} // namespace Activations

} // namespace FJML

#endif