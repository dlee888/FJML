#include <catch2/catch_all.hpp>

#include "../include/FJML/loss.h"
#include "../include/FJML/optimizers.h"

using namespace FJML;
using namespace Catch;

TEST_CASE("Test optimizers in 1d", "[optimizers]") {
    Tensor yhat = Tensor::array({0.9, 0.6, 0.75}), y = Tensor::array(std::vector<float>{1, 0, 1});

    Tensor out_grad = Loss::mse.calc_derivative(y, yhat);
    float orig_loss = Loss::mse.calc_loss(y, yhat);

    SECTION("Testing dummy optimizer") {
        Optimizers::Optimizer opt;
        opt.apply_grad(yhat, out_grad);
        opt.clone();
    }

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

    SECTION("Testing clone") {
        Optimizers::SGD sgd;
        Optimizers::SGD sgd_clone = *((Optimizers::SGD*)sgd.clone());
        REQUIRE(sgd.alpha == sgd_clone.alpha);

        Optimizers::Adam adam;
        Optimizers::Adam adam_clone = *((Optimizers::Adam*)adam.clone());
        REQUIRE(adam.alpha == adam_clone.alpha);
        REQUIRE(adam.beta1 == adam_clone.beta1);
        REQUIRE(adam.beta2 == adam_clone.beta2);
    }
}

TEST_CASE("Testing optimizers in 2d", "[optimizers]") {
    Tensor yhat = Tensor::array(std::vector<std::vector<float>>{{0.9, 0.6}, {0.75, 0.5}}),
           y = Tensor::array(std::vector<std::vector<float>>{{1, 0}, {0.3, 0.6}});

    Tensor out_grad = Loss::mse.calc_derivative(y, yhat);
    REQUIRE(out_grad.shape[0] == 2);
    REQUIRE(out_grad.shape[1] == 2);

    float orig_loss = Loss::mse.calc_loss(y, yhat);

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
