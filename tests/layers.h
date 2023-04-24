#include <catch2/catch_all.hpp>

#include "../src/FJML/layers/layers.h"

using namespace FJML;
using namespace Catch;

TEST_CASE("Test layers", "[layers]") {
    Tensor<double> input = Tensor<double>::array({1, 2, -1});

    SECTION("Test Dense layer") {
        Layers::Dense dense{3, 2, Activations::linear, new Optimizers::SGD{0.1}};

        REQUIRE(dense.weights.shape == std::vector<int>{3, 2});
        REQUIRE(dense.bias.shape == std::vector<int>{2});

        dense.weights.at(0, 0) = 1;
        dense.weights.at(0, 1) = 2;
        dense.weights.at(1, 0) = 3;
        dense.weights.at(1, 1) = 4;
        dense.weights.at(2, 0) = 5;
        dense.weights.at(2, 1) = 6;

        dense.bias.at(0) = -4.20;
        dense.bias.at(1) = 6.9;

        SECTION("Test apply") {
            Tensor<double> output = dense.apply(input);
            REQUIRE(output.shape == std::vector<int>{2});
            REQUIRE(output.at(0) == Approx(-2.2).margin(0.00001));
            REQUIRE(output.at(1) == Approx(10.9).margin(0.00001));

            std::vector<Tensor<double>> vector_input;
            vector_input.push_back(input);
            vector_input.push_back(input);
            vector_input.push_back(input);

            std::vector<Tensor<double>> vector_output = dense.apply(vector_input);
            REQUIRE(vector_output.size() == 3);
            REQUIRE(vector_output.at(0).shape[0] == 2);
            REQUIRE(vector_output.at(1).shape[0] == 2);
            REQUIRE(vector_output.at(2).shape[0] == 2);
        }

        SECTION("Test backward") {
            Tensor<double> output = dense.apply(input);
            std::vector<Tensor<double>> grad = {Tensor<double>::array({1, 2})};

            std::vector<Tensor<double>> input_grad = dense.backward({input}, {dense.apply(input)}, grad);
            REQUIRE(input_grad.at(0).shape == std::vector<int>{3});
            REQUIRE(input_grad.at(0).at(0) == Approx(5));
            REQUIRE(input_grad.at(0).at(1) == Approx(11));
            REQUIRE(input_grad.at(0).at(2) == Approx(17));

            Tensor<double> new_output = dense.apply(input);
            REQUIRE(new_output.at(0) != output.at(0));
            REQUIRE(new_output.at(1) != output.at(1));

            REQUIRE(dense.weights.at(0, 0) == Approx(0.9));
            REQUIRE(dense.weights.at(0, 1) == Approx(1.8));
            REQUIRE(dense.weights.at(1, 0) == Approx(2.8));
            REQUIRE(dense.weights.at(1, 1) == Approx(3.6));
            REQUIRE(dense.weights.at(2, 0) == Approx(5.1));
            REQUIRE(dense.weights.at(2, 1) == Approx(6.2));
        }
    }

    SECTION("Test softmax layer") {
        Layers::Softmax softmax;

        SECTION("Test apply") {
            Tensor<double> output = softmax.apply(input);
            REQUIRE(output.shape == std::vector<int>{3});
            REQUIRE(output.at(0) == Approx(0.25949648));
            REQUIRE(output.at(1) == Approx(0.70538455));
            REQUIRE(output.at(2) == Approx(0.03511903));
        }

        SECTION("Test backward") {
            std::vector<Tensor<double>> grad = {Tensor<double>::array({1, 2, 3})};

            std::vector<Tensor<double>> input_grad = softmax.backward({input}, {softmax.apply(input)}, grad);
            REQUIRE(input_grad.size() == 1);
            REQUIRE(input_grad.at(0).shape == std::vector<int>{3});
            REQUIRE(input_grad.at(0).at(0) == Approx(-0.2013).margin(0.001));
            REQUIRE(input_grad.at(0).at(1) == Approx(0.1583).margin(0.001));
            REQUIRE(input_grad.at(0).at(2) == Approx(0.0430).margin(0.001));
        }
    }
}
