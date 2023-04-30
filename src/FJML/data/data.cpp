// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include "../../../include/FJML/data.h"

namespace FJML {

namespace Data {

Tensor<double> one_hot(int x, int n) {
    Tensor<double> res({n});
    res.at(x) = 1;
    return res;
}

} // namespace Data

} // namespace FJML

