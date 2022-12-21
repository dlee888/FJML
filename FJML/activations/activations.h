#ifndef ACTIVATIONS_INCLUDED
#define ACTIVATIONS_INCLUDED

#include <cmath>
#include <functional>

#include "../util/types.h"

namespace FJML {

/**
 * @brief Activation functions
 *
 * @details Activation functions are used to transform the output of a neuron. They are implemented as a pair of
 * functors, one for the function itself and one for its derivative.
 */
namespace Activations {

/**
 * @brief This class represents a generic activation function
 *
 * @details This class represents a generic activation function. It is implemented as a pair of functors, one for the
 * function itself and one for its derivative.
 */
class Activation {
  public:
	std::function<double(double)> func, grad;
	std::string name;

	Activation() = default;

	/**
	 * @brief Construct a new Activation object
	 * @param _func The function itself
	 * @param _grad The derivative of the function
	 * @param _name The name of the function
	 */
	Activation(std::function<double(double)> _func, std::function<double(double)> _grad, std::string _name)
		: func{_func}, grad{_grad}, name{_name} {}

	/**
	 * @brief apply the function to a layer
	 * @param layer The layer to apply the function to
	 */
	template <int N> void apply(Tensor<N>& layer) {
		for (auto& i : layer) {
			i = apply(i);
		}
	}
};

template <> void Activation::apply(Tensor<1>& layer);

extern Activation linear, sigmoid, swish, relu, leaky_relu, tanh;

extern std::vector<Activation> all_activations;

} // namespace Activations

} // namespace FJML

#endif
