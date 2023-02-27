#include <catch2/catch_all.hpp>

#include <FJML.h>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test optimizers in 1d", "[optimizers]") {
    FJML::Tensor<1> yhat{3}, y{3};
    yhat[0] = 0.9;
    yhat[1] = 0.6;
    yhat[2] = 0.75;

    y[0] = 1;
    y[1] = 0;
    y[2] = 1;

    FJML::Tensor<1> out_grad = FJML::Loss::mse.calc_grad(y, yhat);
    double orig_loss = FJML::Loss::mse.calc_loss(y, yhat);

    SECTION("Testing SGD") {
        FJML::Optimizers::SGD<1> sgd;
        sgd.apply_grad(yhat, out_grad);
        REQUIRE(FJML::Loss::mse.calc_loss(y, yhat) < orig_loss);
    }

    SECTION("Testing Adam") {
        FJML::Optimizers::Adam<1> adam;
        adam.apply_grad(yhat, out_grad);
        REQUIRE(FJML::Loss::mse.calc_loss(y, yhat) < orig_loss);
    }
}

TEST_CASE("Testing optimizers in 2d", "[optimizers]") {
    FJML::Tensor<2> yhat{{2, 2}}, y{{2, 2}};

    yhat[0][0] = 0.9;
    yhat[0][1] = 0.6;
    yhat[1][0] = 0.75;
    yhat[1][1] = 0.5;

    y[0][0] = 1;
    y[0][1] = 0;
    y[1][0] = 0.3;
    y[1][1] = 0.6;

    FJML::Tensor<2> out_grad = FJML::Loss::mse.calc_grad(y, yhat);
    REQUIRE(out_grad.shape[0] == 2);
    REQUIRE(out_grad.shape[1] == 2);

    double orig_loss = FJML::Loss::mse.calc_loss(y, yhat);

    SECTION("Testin SGD") {
        FJML::Optimizers::SGD<2> sgd;
        sgd.apply_grad(yhat, out_grad);
        REQUIRE(FJML::Loss::mse.calc_loss(y, yhat) < orig_loss);
    }

    SECTION("Testing Adam") {
        FJML::Optimizers::Adam<2> adam;
        adam.apply_grad(yhat, out_grad);
        REQUIRE(FJML::Loss::mse.calc_loss(y, yhat) < orig_loss);
    }
}
