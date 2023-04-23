#include <catch2/catch_all.hpp>

#include "../src/FJML/activations/activations.h"

using namespace Catch;
using namespace FJML;

TEST_CASE("Test activations", "[activations]") {
    Tensor<double> x = Tensor<double>::array({1, 2, -1});

    SECTION("Test sigmoid") {
        Tensor<double> y = Activations::sigmoid.forward(x);

        REQUIRE(y.at(0) == Approx(0.7310585786300049));
        REQUIRE(y.at(1) == Approx(0.8807970779778823));
        REQUIRE(y.at(2) == Approx(0.2689414213699951));

        Tensor<double> dy = Activations::sigmoid.backward(x);

        REQUIRE(dy.at(0) == Approx(0.19661193324148185));
        REQUIRE(dy.at(1) == Approx(0.10499358540350662));
        REQUIRE(dy.at(2) == Approx(0.19661193324148185));
    }

    SECTION("Test tanh") {
        Tensor<double> y = Activations::tanh.forward(x);

        REQUIRE(y.at(0) == Approx(0.7615941559557649));
        REQUIRE(y.at(1) == Approx(0.9640275800758169));
        REQUIRE(y.at(2) == Approx(-0.7615941559557649));

        Tensor<double> dy = Activations::tanh.backward(x);

        REQUIRE(dy.at(0) == Approx(0.41997434161402614));
        REQUIRE(dy.at(1) == Approx(0.07065082485316452));
        REQUIRE(dy.at(2) == Approx(0.41997434161402614));
    }

    SECTION("Test relu") {
        Tensor<double> y = Activations::relu.forward(x);

        REQUIRE(y.at(0) == Approx(1));
        REQUIRE(y.at(1) == Approx(2));
        REQUIRE(y.at(2) == Approx(0));

        Tensor<double> dy = Activations::relu.backward(x);

        REQUIRE(dy.at(0) == Approx(1));
        REQUIRE(dy.at(1) == Approx(1));
        REQUIRE(dy.at(2) == Approx(0));
    }

    SECTION("Test leaky relu") {
        Tensor<double> y = Activations::leaky_relu.forward(x);

        REQUIRE(y.at(0) == Approx(1));
        REQUIRE(y.at(1) == Approx(2));
        REQUIRE(y.at(2) == Approx(-0.01));

        Tensor<double> dy = Activations::leaky_relu.backward(x);

        REQUIRE(dy.at(0) == Approx(1));
        REQUIRE(dy.at(1) == Approx(1));
        REQUIRE(dy.at(2) == Approx(0.01));
    }

    SECTION("Test linear") {
        Tensor<double> y = Activations::linear.forward(x);

        REQUIRE(y.at(0) == Approx(1));
        REQUIRE(y.at(1) == Approx(2));
        REQUIRE(y.at(2) == Approx(-1));

        Tensor<double> dy = Activations::linear.backward(x);

        REQUIRE(dy.at(0) == Approx(1));
        REQUIRE(dy.at(1) == Approx(1));
        REQUIRE(dy.at(2) == Approx(1));
    }

    SECTION("Test swish") {
        Tensor<double> y = Activations::swish.forward(x);

        REQUIRE(y.at(0) == Approx(0.7310585786300049));
        REQUIRE(y.at(1) == Approx(1.7615941559557649));
        REQUIRE(y.at(2) == Approx(-0.2689414213699951));

        Tensor<double> dy = Activations::swish.backward(x);

        REQUIRE(dy.at(0) == Approx(0.9276705384254456));
        REQUIRE(dy.at(1) == Approx(1.0907841920852661));
        REQUIRE(dy.at(2) == Approx(0.07232949137687683));
    }
}
