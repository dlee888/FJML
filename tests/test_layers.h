#include <catch2/catch_all.hpp>

#include "../include/FJML/layers.h"

using namespace FJML;
using namespace Catch;

TEST_CASE("Test layers", "[layers]") {
    Tensor input = Tensor::array(std::vector<float>{1, 2, -1});
    input.reshape({1, 3});

    SECTION("Test dummy layer") {
        Layers::Layer layer;

        REQUIRE_NOTHROW(layer.apply(input));
        REQUIRE_NOTHROW(layer.backward(input, input));
        std::ofstream file("/tmp/dummy.fjml");
        REQUIRE_NOTHROW(layer.save(file));
        REQUIRE_NOTHROW(layer.summary());

        std::ofstream file2("/tmp/dummy2.fjml");
        file2 << "Dummy" << std::endl;
        std::ifstream file3("/tmp/dummy2.fjml");
        REQUIRE_THROWS(Layers::load(file3));
    }

    SECTION("Test Dense layer") {
        Layers::Dense dense{3, 2, Activations::linear};
        dense.set_optimizer(new Optimizers::SGD(0.1));

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
            Tensor output = dense.apply(input);
            REQUIRE(output.shape == std::vector<int>{1, 2});
            REQUIRE(output.at(0, 0) == Approx(-2.2).margin(0.00001));
            REQUIRE(output.at(0, 1) == Approx(10.9).margin(0.00001));

            Tensor vector_input = Tensor::array(std::vector<std::vector<float>>{{1, 2, -1}, {1, 2, -1}, {1, 2, -1}});

            Tensor vector_output = dense.apply(vector_input);
            REQUIRE(vector_output.shape[0] == 3);
            REQUIRE(vector_output.shape[1] == 2);
            REQUIRE(vector_output.at(0, 0) == Approx(-2.2).margin(0.00001));
            REQUIRE(vector_output.at(0, 1) == Approx(10.9).margin(0.00001));
            REQUIRE(vector_output.at(1, 0) == Approx(-2.2).margin(0.00001));
            REQUIRE(vector_output.at(1, 1) == Approx(10.9).margin(0.00001));
            REQUIRE(vector_output.at(2, 0) == Approx(-2.2).margin(0.00001));
            REQUIRE(vector_output.at(2, 1) == Approx(10.9).margin(0.00001));
        }

        SECTION("Test backward") {
            Tensor output = dense.apply(input);
            Tensor grad = Tensor::array(std::vector<float>{1, 2});
            grad.reshape({1, 2});

            Tensor input_grad = dense.backward(input, grad);
            REQUIRE(input_grad.shape == std::vector<int>{1, 3});
            REQUIRE(input_grad.at(0, 0) == Approx(5));
            REQUIRE(input_grad.at(0, 1) == Approx(11));
            REQUIRE(input_grad.at(0, 2) == Approx(17));

            Tensor new_output = dense.apply(input);
            REQUIRE(new_output.at(0, 0) != output.at(0, 0));
            REQUIRE(new_output.at(0, 1) != output.at(0, 1));

            REQUIRE(dense.weights.at(0, 0) == Approx(0.9));
            REQUIRE(dense.weights.at(0, 1) == Approx(1.8));
            REQUIRE(dense.weights.at(1, 0) == Approx(2.8));
            REQUIRE(dense.weights.at(1, 1) == Approx(3.6));
            REQUIRE(dense.weights.at(2, 0) == Approx(5.1));
            REQUIRE(dense.weights.at(2, 1) == Approx(6.2));
        }

        SECTION("Test save and load") {
            std::ofstream file("/tmp/dense.fjml");
            dense.save(file);

            std::ifstream file2("/tmp/dense.fjml");
            Layers::Layer* new_layer = Layers::load(file2);
            REQUIRE(new_layer->name == "Dense");
            Layers::Dense new_dense = (Layers::Dense&)*new_layer;

            REQUIRE(new_dense.weights.shape == std::vector<int>{3, 2});
            REQUIRE(new_dense.bias.shape == std::vector<int>{2});

            REQUIRE(new_dense.weights.at(0, 0) == Approx(1));
            REQUIRE(new_dense.weights.at(0, 1) == Approx(2));
            REQUIRE(new_dense.weights.at(1, 0) == Approx(3));
            REQUIRE(new_dense.weights.at(1, 1) == Approx(4));
            REQUIRE(new_dense.weights.at(2, 0) == Approx(5));
            REQUIRE(new_dense.weights.at(2, 1) == Approx(6));

            REQUIRE(new_dense.bias.at(0) == Approx(-4.20));
            REQUIRE(new_dense.bias.at(1) == Approx(6.9));

            std::ofstream file3("/tmp/dense2.fjml");
            file3 << "Dense\nunknown_activation\n3 2\n1 2\n3 4\n5 6\n-4.2 6.9" << std::endl;
            std::ifstream file4("/tmp/dense2.fjml");
            REQUIRE_THROWS(Layers::load(file4));

            std::ofstream file5("/tmp/dense3.fjml");
            file5 << "Dense\nlinear\n3 -2\n1 2\n3 4\n5 6\n-4.2" << std::endl;
            std::ifstream file6("/tmp/dense3.fjml");
            REQUIRE_THROWS(Layers::load(file6));
        }

        SECTION("Test summary") { dense.summary(); }
    }

    SECTION("Test softmax layer") {
        Layers::Softmax softmax;

        SECTION("Test apply") {
            Tensor output = softmax.apply(input);
            REQUIRE(output.shape == std::vector<int>{1, 3});
            REQUIRE(output.at(0, 0) == Approx(0.25949648));
            REQUIRE(output.at(0, 1) == Approx(0.70538455));
            REQUIRE(output.at(0, 2) == Approx(0.03511903));
        }

        SECTION("Test backward") {
            Tensor grad = Tensor::array(std::vector<float>{1, 2, 3});
            grad.reshape({1, 3});

            Tensor input_grad = softmax.backward({input}, grad);
            REQUIRE(input_grad.shape == std::vector<int>{1, 3});
            REQUIRE(input_grad.at(0, 0) == Approx(-0.2013).margin(0.001));
            REQUIRE(input_grad.at(0, 1) == Approx(0.1583).margin(0.001));
            REQUIRE(input_grad.at(0, 2) == Approx(0.0430).margin(0.001));
        }

        SECTION("Test save and load") {
            std::ofstream file("/tmp/softmax.fjml");
            softmax.save(file);

            std::ifstream file2("/tmp/softmax.fjml");
            Layers::Layer* new_layer = Layers::load(file2);
            REQUIRE(new_layer->name == "Softmax");
        }

        SECTION("Test summary") { softmax.summary(); }
    }
}
