#include <catch2/catch_all.hpp>

#include "../include/FJML/mlp.h"

using namespace Catch;
using namespace FJML;

TEST_CASE("Test metrics", "[metrics]") {
    SECTION("Test accuracy") {
        Tensor labels = Tensor::array(std::vector<std::vector<double>>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
        Tensor predictions =
            Tensor::array(std::vector<std::vector<double>>{{1, 0, 0}, {0.1, 0.8, 0.1}, {0.6, 0.3, 0.1}});

        REQUIRE(MLP::accuracy.compute(labels, predictions) == Approx(2.0 / 3.0));
    }

    SECTION("Test sparse accuracy") {
        Tensor labels = Tensor::array(std::vector<double>{0, 1, 2});
        Tensor predictions =
            Tensor::array(std::vector<std::vector<double>>{{1, 0, 0}, {0.1, 0.8, 0.1}, {0.6, 0.3, 0.1}});

        REQUIRE(MLP::sparse_categorical_accuracy.compute(labels, predictions) == Approx(2.0 / 3.0));
    }

    SECTION("Test mean squared error") {
        Tensor labels = Tensor::array(std::vector<std::vector<double>>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
        Tensor predictions =
            Tensor::array(std::vector<std::vector<double>>{{1, 0, 0}, {0.1, 0.8, 0.1}, {0.6, 0.3, 0.1}});

        REQUIRE(MLP::mean_squared_error.compute(labels, predictions) == Approx(0.14666666666666666666666666666667));
    }
}
