#include <FJML/linalg/tensor.h>
#include <catch2/catch_all.hpp>

#include <FJML.h>

TEST_CASE("Testing loss functions", "[loss]") {
    FJML::Tensor<1> y{3}, yhat{3};
    y[0] = 1;
    y[1] = 2;
    y[2] = 3;
    yhat[0] = 3;
    yhat[1] = 2;
    yhat[2] = 1;

    SECTION("Testing MSE") {
        REQUIRE(FJML::Loss::mse.calc_loss(y, yhat) == 8);

        FJML::Tensor<1> dy = FJML::Loss::mse.calc_grad(y, yhat);
        REQUIRE(dy[0] == 4);
        REQUIRE(dy[1] == 0);
        REQUIRE(dy[2] == -4);
    }

    SECTION("Testing huber") {
        REQUIRE(FJML::Loss::huber.calc_loss(y, yhat) == 3);

        FJML::Tensor<1> dy = FJML::Loss::huber.calc_grad(y, yhat);
        REQUIRE(dy[0] == 1);
        REQUIRE(dy[1] == 0);
        REQUIRE(dy[2] == -1);
    }
}
