#ifndef LAYER_INCLUDED
#define LAYER_INCLUDED

#include <cassert>
#include <chrono>
#include <climits>
#include <fstream>
#include <iostream>
#include <random>

#include "../activations/activations.h"
#include "../optimizers/optimizers.h"
#include "../util/linalg.h"
#include "../util/types.h"

namespace FJML {

namespace Layers {

// Based class - does nothing, meant to be inherited
class Layer {
  public:
	std::string name;

	Layer() {}
	Layer(std::string _name) : name{_name} {}
	virtual ~Layer() {}

	virtual layer_vals apply(const layer_vals& input);
	virtual std::vector<layer_vals> apply(const std::vector<layer_vals>& input);

	// Return the gradient w.r.t the input nodes of this layer
	virtual std::vector<layer_vals> apply_grad(const std::vector<layer_vals>& input_vals,
											   const std::vector<layer_vals>& output_grad);

	virtual void save(std::ofstream& file);

	virtual void summary() {}
};

class Dense : public Layer {
  public:
	int input_size, output_size;
	weights layer_weights;
	bias layer_bias;
	Activations::Activation activ;
	Optimizers::Optimizer<2>* w_opt;
	Optimizers::Optimizer<1>* b_opt;

	Dense();
	Dense(int _input, int _output, Activations::Activation a = Activations::sigmoid, bool randomize = true,
		  Optimizers::Optimizer<1>* o = new Optimizers::SGD<1>);
	Dense(std::ifstream& file);
	~Dense() {}

	layer_vals apply(const layer_vals& input);
	std::vector<layer_vals> apply(const std::vector<layer_vals>& input);
	std::vector<layer_vals> apply_grad(const std::vector<layer_vals>& input_vals,
									   const std::vector<layer_vals>& output_grad);

	void save(std::ofstream& file);

	void summary();
};

class Softmax : public Layer {
	void norm(layer_vals& input);

  public:
	Softmax() { name = "Softmax"; }
	~Softmax() {}

	layer_vals apply(const layer_vals& input);
	std::vector<layer_vals> apply(const std::vector<layer_vals>& input);
	std::vector<layer_vals> apply_grad(const std::vector<layer_vals>& input_vals,
									   const std::vector<layer_vals>& output_grad);

	void save(std::ofstream& file);

	void summary();
};

Layer* load(std::ifstream& file);

} // namespace Layers

} // namespace FJML

#endif