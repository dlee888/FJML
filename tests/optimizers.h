#include <catch2/catch_all.hpp>

#include "../include/FJML/loss.h"
#include "../include/FJML/optimizers.h"

using namespace FJML;
using namespace Catch;

TEST_CASE("Test optimizers in 1d", "[optimizers]") {
    Tensor<double> yhat = Tensor<double>::array({0.9, 0.6, 0.75}), y = Tensor<double>::array({1, 0, 1});

    Tensor<double> out_grad = Loss::mse.calc_derivative(y, yhat);
    double orig_loss = Loss::mse.calc_loss(y, yhat);

    SECTION("Testing SGD") {
        Optimizers::SGD sgd;
        sgd.apply_grad(yhat, out_grad);
        REQUIRE(Loss::mse.calc_loss(y, yhat) < orig_loss);
    }

    SECTION("Testing Adam") {
        Optimizers::Adam adam;
        adam.apply_grad(yhat, out_grad);
        REQUIRE(Loss::mse.calc_loss(y, yhat) < orig_loss);
    }
}

TEST_CASE("Testing optimizers in 2d", "[optimizers]") {
    Tensor<double> yhat = Tensor<double>::array(std::vector<std::vector<double>>{{0.9, 0.6}, {0.75, 0.5}}),
                   y = Tensor<double>::array(std::vector<std::vector<double>>{{1, 0}, {0.3, 0.6}});

    Tensor<double> out_grad = Loss::mse.calc_derivative(y, yhat);
    REQUIRE(out_grad.shape[0] == 2);
    REQUIRE(out_grad.shape[1] == 2);

    double orig_loss = Loss::mse.calc_loss(y, yhat);

    SECTION("Testin SGD") {
        Optimizers::SGD sgd;
        sgd.apply_grad(yhat, out_grad);
        REQUIRE(Loss::mse.calc_loss(y, yhat) < orig_loss);
    }

    SECTION("Testing Adam") {
        Optimizers::Adam adam;
        adam.apply_grad(yhat, out_grad);
        REQUIRE(Loss::mse.calc_loss(y, yhat) < orig_loss);
    }
}
