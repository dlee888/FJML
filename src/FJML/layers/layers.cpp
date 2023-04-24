// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include "layers.h"

namespace FJML {

namespace Layers {

Layer* load(std::ifstream& file) {
    std::string type;
    file >> type;
    if (type == "Dense") {
        return new Layers::Dense(file);
    }
    if (type == "Softmax") {
        return new Layers::Softmax;
    }
    throw std::runtime_error("Invalid layer type");
}

} // namespace Layers

} // namespace FJML
