#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include "FJML.h"

using namespace FJML;
using namespace Catch;

TEST_CASE("Test layers", "[layers]") {
    layer_vals input{3};
    input[0] = 1;
    input[1] = 2;
    input[2] = -1;

    SECTION("Test Dense layer") {
        Layers::Dense dense{3, 2, Activations::sigmoid, new Optimizers::SGD<1>{0.1}};

        SECTION("Test apply") {
            layer_vals output = dense.apply(input);
            REQUIRE(output.size() == 2);

            std::vector<layer_vals> vector_input;
            vector_input.push_back(input);
            vector_input.push_back(input);
            vector_input.push_back(input);

            std::vector<layer_vals> vector_output = dense.apply(vector_input);
            REQUIRE(vector_output.size() == 3);
            REQUIRE(vector_output[0].size() == 2);
            REQUIRE(vector_output[1].size() == 2);
            REQUIRE(vector_output[2].size() == 2);
        }

        SECTION("Test apply_grad") {
            layer_vals output = dense.apply(input);
            std::vector<layer_vals> grad;
            grad.push_back(Tensor<1>{2});
            grad[0][0] = 1;
            grad[0][1] = 2;

            std::vector<layer_vals> input_grad = dense.apply_grad({input}, grad);
            REQUIRE(input_grad[0].size() == 3);

            layer_vals new_output = dense.apply(input);
            REQUIRE(new_output[0] != output[0]);
            REQUIRE(new_output[1] != output[1]);
        }
    }

    SECTION("Test softmax layer") {
        Layers::Softmax softmax;

        SECTION("Test apply") {
            layer_vals output = softmax.apply(input);
            REQUIRE(output.size() == 3);
            REQUIRE(output[0] == Approx(0.25949648));
            REQUIRE(output[1] == Approx(0.70538455));
            REQUIRE(output[2] == Approx(0.03511903));
        }

        SECTION("Test apply_grad") {
            layer_vals output = softmax.apply(input);
            REQUIRE(output.size() == 3);

            std::vector<layer_vals> grad;
            grad.push_back(Tensor<1>{3});
            grad[0][0] = 1;
            grad[0][1] = 2;
            grad[0][2] = 3;

            std::vector<layer_vals> input_grad = softmax.apply_grad({input}, grad);
            REQUIRE(input_grad[0].size() == 3);
        }
    }
}
