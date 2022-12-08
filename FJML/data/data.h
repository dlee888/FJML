#ifndef DATA_INCLUDED
#define DATA_INCLUDED

#include <algorithm>
#include <climits>
#include <fstream>
#include <sstream>

#include "../util/types.h"

namespace FJML {

namespace Data {

layer_vals one_hot(int x, int n) {
	layer_vals res({n});
	res[x] = 1;
	return res;
}

void load_mnist(int k, std::string filename, std::vector<layer_vals>& input_set, std::vector<layer_vals>& output_set,
				int n = INT_MAX) {
	std::string temp;
	std::ifstream data_file(filename);
	if (!data_file.is_open()) {
		std::cout << filename << " failed to open." << std::endl;
	}

	std::string line;
	while (std::getline(data_file, line)) {
		std::stringstream ss(line);

		layer_vals input_item;
		for (int i = 0; i < k; i++) {
			std::getline(ss, temp, ',');
			input_item.push_back(std::stod(temp) / 255);
		}
		input_set.push_back(input_item);

		std::getline(ss, temp, ',');
		output_set.push_back(one_hot(std::stoi(temp), 10));

		if ((int)input_set.size() >= n) {
			break;
		}
	}
}

void split(std::vector<layer_vals>& input_set, std::vector<layer_vals>& output_set,
		   std::vector<layer_vals>& input_train, std::vector<layer_vals>& output_train,
		   std::vector<layer_vals>& input_test, std::vector<layer_vals>& output_test, double train_frac = 0.8) {
	int n = input_set.size();
	int train_n = n * train_frac;

	std::vector<int> indices(n);
	for (int i = 0; i < n; i++) {
		indices[i] = i;
	}
	std::random_shuffle(indices.begin(), indices.end());

	for (int i = 0; i < train_n; i++) {
		input_train.push_back(input_set[indices[i]]);
		output_train.push_back(output_set[indices[i]]);
	}
	for (int i = train_n; i < n; i++) {
		input_test.push_back(input_set[indices[i]]);
		output_test.push_back(output_set[indices[i]]);
	}
}

} // namespace Data

} // namespace FJML

#endif