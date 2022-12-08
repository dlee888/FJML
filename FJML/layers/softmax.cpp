#include <cmath>

#include "layers.h"

namespace FJML {

namespace Layers {

void Softmax::norm(layer_vals& input) {
	double max = input[0];
	for (int i = 1; i < (int)input.size(); i++) {
		if (input[i] > max) {
			max = input[i];
		}
	}
	for (int i = 0; i < (int)input.size(); i++) {
		input[i] -= max;
	}
}

layer_vals Softmax::apply(const layer_vals& input) {
	layer_vals res = input;
	norm(res);
	double sum = 0;

	for (double& d : res) {
		d = exp(d);
		sum += d;
	}
	for (double& d : res) {
		d /= sum;
	}

	return res;
}

std::vector<layer_vals> Softmax::apply(const std::vector<layer_vals>& input) {
	std::vector<layer_vals> res;
	for (const layer_vals& l : input) {
		res.push_back(apply(l));
	}
	return res;
}

std::vector<layer_vals> Softmax::apply_grad(const std::vector<layer_vals>& input_vals,
											const std::vector<layer_vals>& output_grad) {
	assert(input_vals.size() == output_grad.size());
	assert(input_vals[0].size() == output_grad[0].size());
	int n = input_vals.size(), m = input_vals[0].size();

	std::vector<layer_vals> res(n, layer_vals{{(int)input_vals[0].size()}});
	for (int i = 0; i < n; i++) {
		layer_vals out = input_vals[i];
		norm(out);

		double sum = 0;
		for (double& d : out) {
			d = exp(d);
			sum += d;
		}

		double denom = sum * sum;
		for (int j = 0; j < m; j++) {
			for (int k = 0; k < m; k++) {
				res[i][j] += output_grad[i][k] * (out[j] * (k == j ? sum - out[j] : -out[k])) / denom;
			}
		}
	}

	return res;
}

void Softmax::save(std::ofstream& file) { file << "Softmax" << std::endl; }

void Softmax::summary() { std::cout << "Softmax layer" << std::endl; }

} // namespace Layers

} // namespace FJML