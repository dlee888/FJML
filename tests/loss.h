#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include <FJML.h>

using namespace FJML;
using namespace Catch;

TEST_CASE("Testing loss functions", "[loss]") {
    Tensor<1> y{3}, yhat{3};
    y[0] = 1;
    y[1] = 2;
    y[2] = 3;
    yhat[0] = 3;
    yhat[1] = 2;
    yhat[2] = 1;

    SECTION("Testing MSE") {
        REQUIRE(Loss::mse.calc_loss(y, yhat) == 8);

        Tensor<1> dy = Loss::mse.calc_grad(y, yhat);
        REQUIRE(dy[0] == 4);
        REQUIRE(dy[1] == 0);
        REQUIRE(dy[2] == -4);
    }

    SECTION("Testing huber") {
        REQUIRE(Loss::huber.calc_loss(y, yhat) == 3);

        Tensor<1> dy = Loss::huber.calc_grad(y, yhat);
        REQUIRE(dy[0] == 1);
        REQUIRE(dy[1] == 0);
        REQUIRE(dy[2] == -1);
    }

    SECTION("Testing crossentropy") {
        y[0] = 0;
        y[1] = 0;
        y[2] = 1;
        yhat[0] = 0.3;
        yhat[1] = 0.4;
        yhat[2] = 0.3;

        REQUIRE(Loss::crossentropy.calc_loss(y, yhat) == Approx(2.0714733720306593));
    }
}
