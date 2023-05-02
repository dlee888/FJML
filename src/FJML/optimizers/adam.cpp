// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <cmath>

#include "../../../include/FJML/optimizers.h"

namespace FJML {

namespace Optimizers {

void Adam::init(const Tensor<double>& params) {
    if (m.shape != params.shape) {
        m = Tensor<double>(params.shape);
        v = Tensor<double>(params.shape);
        t = 1;
    }
}

void Adam::apply_grad(Tensor<double>& params, const Tensor<double>& grads) {
    init(params);
    // Update the first and second moment estimates
    m = beta1 * m + (1 - beta1) * grads;
    v = beta2 * v + (1 - beta2) * grads * grads;

    // Compute bias-corrected first and second moment estimates
    Tensor<double> m_hat = m / (1 - pow(beta1, t));
    Tensor<double> v_hat = v / (1 - pow(beta2, t));

    // Update parameters
    params -= alpha * m_hat / (v_hat.apply_function([](double x) { return std::sqrt(x); }) + epsilon);

    // Increment time step
    t++;
}

} // namespace Optimizers

} // namespace FJML
