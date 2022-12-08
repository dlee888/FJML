#ifndef MLP_INCLUDED
#define MLP_INCLUDED

#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "../layers/layers.h"
#include "../loss/loss.h"
#include "../optimizers/optimizers.h"
#include "../util/linalg.h"
#include "../util/types.h"

namespace FJML {

class MLP {
  public:
	std::vector<Layers::Layer*> layers;
	Loss::Loss loss_fn;

	MLP() {}
	MLP(int input_size, int output_size, std::vector<int> hidden_sizes = std::vector<int>(),
		std::vector<Activations::Activation> _activations = std::vector<Activations::Activation>(),
		Loss::Loss _loss = Loss::mse, Optimizers::Optimizer<1>* optimizer = new Optimizers::SGD<1>);
	MLP(std::vector<Layers::Layer*> _layers, Loss::Loss _loss,
		Optimizers::Optimizer<1>* optimizer = new Optimizers::SGD<1>)
		: layers{_layers}, loss_fn{_loss} {
		set_optimizer(optimizer);
	}
	~MLP() {
		for (Layers::Layer* l : layers) {
			delete l;
		}
	}

	void set_loss(Loss::Loss l) { loss_fn = l; }
	void set_optimizer(Optimizers::Optimizer<1>* o) {
		for (Layers::Layer* l : layers) {
			if (l->name == "Dense") {
				((Layers::Dense*)l)->w_opt = Optimizers::get_optimizer<2>(o);
				((Layers::Dense*)l)->b_opt = Optimizers::get_optimizer<1>(o);
			}
		}
	}

	void add(Layers::Layer l) { layers.push_back(&l); }
	void add(Layers::Layer* l) { layers.push_back(l); }

	layer_vals run(layer_vals input);

	double calc_loss(std::vector<layer_vals> x_test, std::vector<layer_vals> y_test);
	double calc_accuracy(std::vector<layer_vals> x_test, std::vector<layer_vals> y_test);

	void grad_descent(std::vector<layer_vals> x_train, std::vector<layer_vals> y_train,
					  std::vector<std::vector<bool>>* mask = nullptr);

	void save(std::string filename);
	void load(std::string filename);

	void train(std::vector<layer_vals> x_train, std::vector<layer_vals> y_train, std::vector<layer_vals> x_test,
			   std::vector<layer_vals> y_test, int epochs, int batch_size, std::string save_file,
			   std::vector<std::vector<bool>>* mask = nullptr);

	void summary();
};

} // namespace FJML

#endif
