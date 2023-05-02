// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include "../../../include/FJML/optimizers.h"

namespace FJML {

namespace Optimizers {

void SGD::apply_grad(Tensor<double>& params, const Tensor<double>& grads) { params -= grads * alpha; }

} // namespace Optimizers

} // namespace FJML
