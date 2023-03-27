#include <cmath>

#include "layers.h"

namespace FJML {

namespace Layers {

layer_vals Softmax::norm(const layer_vals& input) const {
    layer_vals ret(input.size());
    double max = input[0];
    for (int i = 1; i < (int)input.size(); i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }
    for (int i = 0; i < (int)input.size(); i++) {
        ret[i] = input[i] - max;
    }
    return ret;
}

layer_vals Softmax::apply(const layer_vals& input) const {
    layer_vals res = norm(input);
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

std::vector<layer_vals> Softmax::apply(const std::vector<layer_vals>& input) const {
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

    std::vector<layer_vals> res(n, layer_vals{m});
    for (int i = 0; i < n; i++) {
        layer_vals out = norm(input_vals[i]);

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

void Softmax::save(std::ofstream& file) const { file << "Softmax" << std::endl; }

void Softmax::summary() const { std::cout << "Softmax layer" << std::endl; }

} // namespace Layers

} // namespace FJML
