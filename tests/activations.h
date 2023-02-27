#include <catch2/catch_all.hpp>

#include <FJML.h>

using namespace Catch;

TEST_CASE("Test activations", "[activations]") {
    FJML::Tensor<1> x{3};
    x[0] = 1;
    x[1] = 2;
    x[2] = -1;

    SECTION("Test sigmoid") {
        FJML::Tensor<1> y = FJML::Activations::sigmoid.apply(x);
        REQUIRE(y[0] == Approx(0.7310585786300049));
        REQUIRE(y[1] == Approx(0.8807970779778823));
        REQUIRE(y[2] == Approx(0.2689414213699951));
    }

    SECTION("Test tanh") {
        FJML::Tensor<1> y = FJML::Activations::tanh.apply(x);
        REQUIRE(y[0] == Approx(0.7615941559557649));
        REQUIRE(y[1] == Approx(0.9640275800758169));
        REQUIRE(y[2] == Approx(-0.7615941559557649));
    }

    SECTION("Test relu") {
        FJML::Tensor<1> y = FJML::Activations::relu.apply(x);
        REQUIRE(y[0] == Approx(1));
        REQUIRE(y[1] == Approx(2));
        REQUIRE(y[2] == Approx(0));
    }

    SECTION("Test leaky relu") {
        FJML::Tensor<1> y = FJML::Activations::leaky_relu.apply(x);
        REQUIRE(y[0] == Approx(1));
        REQUIRE(y[1] == Approx(2));
        REQUIRE(y[2] == Approx(-0.01));
    }

    SECTION("Test linear") {
        FJML::Tensor<1> y = FJML::Activations::linear.apply(x);
        REQUIRE(y[0] == Approx(1));
        REQUIRE(y[1] == Approx(2));
        REQUIRE(y[2] == Approx(-1));
    }

    SECTION("Test swish") {
        FJML::Tensor<1> y = FJML::Activations::swish.apply(x);
        REQUIRE(y[0] == Approx(0.7310585786300049));
        REQUIRE(y[1] == Approx(1.7615941559557649));
        REQUIRE(y[2] == Approx(-0.2689414213699951));
    }
}
