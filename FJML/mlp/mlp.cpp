#include "mlp.h"
#include "../util/util.h"

namespace FJML {

MLP::MLP(int input_size, int output_size, std::vector<int> hidden_sizes,
		 std::vector<Activations::Activation> _activations, Loss::Loss _loss, Optimizers::Optimizer<1>* o) {
	layers.clear();
	if (_activations.size() != hidden_sizes.size() + 1) {
		_activations = std::vector<Activations::Activation>(hidden_sizes.size() + 1, Activations::sigmoid);
	}

	if (hidden_sizes.size() == 0) {
		layers.push_back(new Layers::Dense(input_size, output_size, _activations[0], o));
	} else {
		layers.push_back(new Layers::Dense(input_size, hidden_sizes[0], _activations[0], o));
		for (int i = 0; i < (int)hidden_sizes.size() - 1; i++) {
			layers.push_back(new Layers::Dense(hidden_sizes[i], hidden_sizes[i + 1], _activations[i + 1], o));
		}
		layers.push_back(new Layers::Dense(hidden_sizes[hidden_sizes.size() - 1], output_size,
										   _activations[hidden_sizes.size()], o));
	}
	loss_fn = _loss;
}

void MLP::grad_descent(std::vector<layer_vals> x_train, std::vector<layer_vals> y_train,
					   std::vector<std::vector<bool>>* mask) {
	assert(x_train.size() == y_train.size());
	int num_inputs = x_train.size(), num_layers = layers.size();

	std::vector<std::vector<layer_vals>> run_res(num_layers + 1);
	run_res[0] = x_train;
	for (int i = 0; i < num_layers; i++) {
		run_res[i + 1] = layers[i]->apply(run_res[i]);
	}

	std::vector<layer_vals> out_grad(num_inputs);
	for (int i = 0; i < num_inputs; i++) {
		if (mask != nullptr) {
			out_grad[i] = loss_fn.calc_grad(y_train[i], run_res[num_layers][i], &(mask->at(i)));
		} else {
			out_grad[i] = loss_fn.calc_grad(y_train[i], run_res[num_layers][i]);
		}
	}

	for (int i = num_layers - 1; i >= 0; i--) {
		out_grad = layers[i]->apply_grad(run_res[i], out_grad);
	}
}

layer_vals MLP::run(layer_vals input) {
	for (Layers::Layer* l : layers) {
		input = l->apply(input);
	}
	return input;
}

double MLP::calc_loss(std::vector<layer_vals> x_test, std::vector<layer_vals> y_test) {
	assert(x_test.size() == y_test.size());
	double total = 0;
	for (int i = 0; i < (int)x_test.size(); i++) {
		layer_vals y_pred = run(x_test[i]);
		total += loss_fn.calc_loss(y_test[i], y_pred);
	}
	return total / x_test.size();
}

double MLP::calc_accuracy(std::vector<layer_vals> x_test, std::vector<layer_vals> y_test) {
	assert(x_test.size() == y_test.size());
	int correct = 0;
	for (int i = 0; i < (int)x_test.size(); i++) {
		layer_vals y_pred = run(x_test[i]);
		if (LinAlg::argmax(y_pred) == LinAlg::argmax(y_test[i])) {
			correct++;
		}
	}
	return (double)correct / x_test.size();
}

void MLP::train(std::vector<layer_vals> x_train, std::vector<layer_vals> y_train, std::vector<layer_vals> x_test,
				std::vector<layer_vals> y_test, int epochs, int batch_size, std::string save_file,
				std::vector<std::vector<bool>>* mask) {
	assert(x_train.size() == y_train.size());
	assert(x_test.size() == y_test.size());
	int num_inputs = x_train.size();
	for (int i = 0; i < epochs; i++) {
		std::vector<int> indices(num_inputs);
		std::iota(indices.begin(), indices.end(), 0);
		std::random_shuffle(indices.begin(), indices.end());
		for (int j = 0; j < num_inputs; j += batch_size) {
			std::vector<layer_vals> x_batch, y_batch;
			std::vector<std::vector<bool>> mask_batch;
			for (int k = j; k < std::min(j + batch_size, num_inputs); k++) {
				x_batch.push_back(x_train[indices[k]]);
				y_batch.push_back(y_train[indices[k]]);
				if (mask != nullptr) {
					mask_batch.push_back(mask->at(indices[k]));
				}
			}
			progress_bar(j, num_inputs);
			grad_descent(x_batch, y_batch, mask != nullptr ? &mask_batch : nullptr);
		}
		progress_bar(num_inputs, num_inputs);
		std::cout << std::endl;
		std::cout << "Epoch " << i << ":\n";
		std::cout << "Train loss: " << calc_loss(x_train, y_train) << "\n";
		std::cout << "Train accuracy: " << calc_accuracy(x_train, y_train) << "\n";
		std::cout << "Test loss: " << calc_loss(x_test, y_test) << "\n";
		std::cout << "Test accuracy: " << calc_accuracy(x_test, y_test) << "\n";
		if (save_file != "") {
			save(save_file);
		}
	}
}

void MLP::save(std::string filename) {
	std::ofstream file(filename);
	file << layers.size() << std::endl;
	for (Layers::Layer* l : layers) {
		l->save(file);
	}
}

void MLP::load(std::string filename) {
	std::ifstream file(filename);
	int num_layers;
	file >> num_layers;
	layers.clear();
	for (int i = 0; i < num_layers; i++) {
		layers.push_back(Layers::load(file));
	}
}

void MLP::summary() {
	std::cout << "Layers:\n";
	for (int i = 0; i < (int)layers.size(); i++) {
		std::cout << "Layer " << i << ":\n";
		layers[i]->summary();
	}
}

} // namespace FJML