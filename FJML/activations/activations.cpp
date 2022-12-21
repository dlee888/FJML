#include <cassert>
#include <cmath>

#include "activations.h"

namespace FJML {

namespace Activations {

template <> void Activation::apply(Tensor<1>& layer) {
	for (auto& i : layer) {
		i = func(i);
	}
}

/**
 * @brief The sigmoid function
 *
 * The sigmoid function is defined as:
 * \f[\sigma(x) = \frac{1}{1 + e^{-x}}\f]
 */
Activation sigmoid([](double x) { return 1 / (1 + exp(-x)); },
				   [](double x) -> double {
					   if (abs(x) > 100) {
						   return 0;
					   }
					   double expx = exp(-x);
					   return expx / (1 + expx) / (1 + expx);
				   },
				   "sigmoid");

/**
 * @brief The linear function
 *
 * Simply returns the input.
 */
Activation linear([](double x) { return x; },
				  [](double x) {
					  assert(x != NAN);
					  return 1;
				  },
				  "linear");

/**
 * @brief The swish function
 *
 * The swish function is defined as:
 * \f[\sigma(x) = \frac{x}{1 + e^{-x}}\f]
 */
Activation swish([](double x) { return x / (1 + exp(-x)); },
				 [](double x) {
					 double expx = exp(-x);
					 double denom = 1 + expx;
					 return (1 + expx + x * expx) / denom / denom;
				 },
				 "swish");

/**
 * @brief The relu function
 *
 * The relu function is defined as:
 * \f[\sigma(x) = \max(0, x)\f]
 */
Activation relu([](double x) -> double { return std::max(0.0, x); }, [](double x) -> double { return x > 0 ? 1 : 0; },
				"relu");

/**
 * @brief The leaky relu function
 *
 * The leaky relu function is defined as:
 * \f[\sigma(x) = \max(0.01x, x)\f]
 */
Activation leaky_relu([](double x) -> double { return x > 0 ? x : 0.01 * x; },
					  [](double x) -> double { return x > 0 ? 1 : 0.01; }, "leaky_relu");

/**
 * @brief The tanh function
 * The tanh function is defined as:
 * \f[\sigma(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}\f]
 */
Activation tanh([](double x) { return std::tanh(x); }, [](double x) { return 1 - std::tanh(x) * std::tanh(x); },
				"tanh");

/**
 * @brief A list of all the activations
 */
std::vector<Activation> all_activations = {sigmoid, linear, swish, relu, leaky_relu, tanh};

} // namespace Activations

} // namespace FJML
