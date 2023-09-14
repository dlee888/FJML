// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef METRICS_INCLUDED
#define METRICS_INCLUDED

#include <functional>
#include <string>

#include "tensor.h"

/*
 * This file contains common metrics used to evaluate the performance of the model.
 */
namespace FJML {

namespace MLP {

/**
 * @brief This is class represents a metric.
 */
class Metric {
  public:
    /**
     * @brief This is the name of the metric.
     */
    std::string name;
    /**
     * @brief This function computes the metric.
     *
     * The first argument is the label, and the second argument is the output.
     *
     * Note: the arguments are assumed to be a batch of data.
     */
    std::function<double(const Tensor&, const Tensor&)> compute;

    /**
     * Default constructor
     */
    Metric() = default;

    /**
     * Constructor
     */
    Metric(std::string name, std::function<double(const Tensor&, const Tensor&)> compute);
};

extern Metric accuracy, mean_squared_error, sparse_categorical_accuracy;

} // namespace MLP

} // namespace FJML

#endif
