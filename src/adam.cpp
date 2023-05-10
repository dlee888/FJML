// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <cmath>

#include "../include/FJML/optimizers.h"

namespace FJML {

namespace Optimizers {

void Adam::init(const Tensor& params) {
    if (m.data_size[0] != params.data_size[0] || v.data_size[0] != params.data_size[0]) {
        m = Tensor(params.shape, params.device);
        v = Tensor(params.shape, params.device);
        t = 1;
    }
}

void Adam::apply_grad(Tensor& params, const Tensor& grads) {
    init(params);
    // Update the first and second moment estimates
    m = beta1 * m + (1 - beta1) * grads;
    v = beta2 * v + (1 - beta2) * grads * grads;

    // Compute bias-corrected first and second moment estimates
    Tensor m_hat = m / (1 - pow(beta1, t));
    Tensor v_hat = v / (1 - pow(beta2, t));

    // Update parameters
    params -= alpha * m_hat / (v_hat.apply_function([](double x) { return std::sqrt(x); }) + epsilon);

    // Increment time step
    t++;
}

} // namespace Optimizers

} // namespace FJML
