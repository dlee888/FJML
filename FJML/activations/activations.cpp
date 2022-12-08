#include <cassert>

#include "activations.h"

namespace FJML {

namespace Activations {

Activation sigmoid([](double x) { return 1 / (1 + exp(-x)); },
				   [](double x) -> double {
					   if (abs(x) > 100) {
						   return 0;
					   }
					   double expx = exp(-x);
					   return expx / (1 + expx) / (1 + expx);
				   },
				   "sigmoid");
Activation linear([](double x) { return x; },
				  [](double x) {
					  assert(x != NAN);
					  return 1;
				  },
				  "linear");
Activation swish([](double x) { return x / (1 + exp(-x)); },
				 [](double x) {
					 double expx = exp(-x);
					 double denom = 1 + expx;
					 return (1 + expx + x * expx) / denom / denom;
				 },
				 "swish");
Activation relu([](double x) -> double { return std::max(0.0, x); }, [](double x) -> double { return x > 0 ? 1 : 0; },
				"relu");
Activation leaky_relu([](double x) -> double { return x > 0 ? x : 0.01 * x; },
					  [](double x) -> double { return x > 0 ? 1 : 0.01; }, "leaky_relu");
Activation tanh([](double x) { return std::tanh(x); }, [](double x) { return 1 - std::tanh(x) * std::tanh(x); },
				"tanh");

std::vector<Activation> all_activations = {sigmoid, linear, swish, relu, leaky_relu, tanh};

} // namespace Activations

} // namespace FJML