#include <catch2/catch_all.hpp>

#include "../include/FJML/loss.h"

using namespace FJML;
using namespace Catch;

TEST_CASE("Testing loss functions", "[loss]") {
    Tensor y = Tensor::array(std::vector<double>{1, 2, 3}), yhat = Tensor::array(std::vector<double>{3, 2, 1});

    SECTION("Testing MSE") {
        REQUIRE(Loss::mse.calc_loss(y, yhat) == 8);

        Tensor dy = Loss::mse.calc_derivative(y, yhat);
        REQUIRE(dy.at(0) == 4);
        REQUIRE(dy.at(1) == 0);
        REQUIRE(dy.at(2) == -4);
    }

    SECTION("Testing huber") {
        REQUIRE(Loss::huber.calc_loss(y, yhat) == 6);

        Tensor dy = Loss::huber.calc_derivative(y, yhat);
        REQUIRE(dy.at(0) == 2);
        REQUIRE(dy.at(1) == 0);
        REQUIRE(dy.at(2) == -2);
    }

    SECTION("Testing crossentropy") {
        y.at(0) = 0;
        y.at(1) = 0;
        y.at(2) = 1;
        yhat.at(0) = 0.3;
        yhat.at(1) = 0.4;
        yhat.at(2) = 0.3;

        Loss::Loss loss = Loss::crossentropy(false);

        REQUIRE(loss.calc_loss(y, yhat) == Approx(2.0714733720306593));

        Tensor dy = loss.calc_derivative(y, yhat);

        REQUIRE(dy.at(0) == Approx(1 / 0.7));
        REQUIRE(dy.at(1) == Approx(1 / 0.6));
        REQUIRE(dy.at(2) == Approx(-1 / 0.3));
    }
}
