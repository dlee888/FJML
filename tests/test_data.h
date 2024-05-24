#include <catch2/catch_all.hpp>

#include "../include/FJML/data.h"

using namespace FJML;

TEST_CASE("Testing data", "[data]") {
    SECTION("Testing one hot") {
        Tensor y = Tensor::array(std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        Tensor one_hot = Data::one_hot(y, 10);

        REQUIRE(one_hot.shape[0] == 10);
        REQUIRE(one_hot.shape[1] == 10);

        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                if (i == j) {
                    REQUIRE(one_hot.at(i, j) == 1);
                } else {
                    REQUIRE(one_hot.at(i, j) == 0);
                }
            }
        }

        Tensor y2d = Tensor::array(std::vector<std::vector<float>>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 0, 1}});
        Tensor one_hot_2d = Data::one_hot(y2d, 10);

        REQUIRE(one_hot_2d.shape == std::vector<int>{4, 3, 10});

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 10; k++) {
                    if (k == y2d.at(i, j)) {
                        REQUIRE(one_hot_2d.at(i, j, k) == 1);
                    } else {
                        REQUIRE(one_hot_2d.at(i, j, k) == 0);
                    }
                }
            }
        }
    }

    SECTION("Testing split") {
        Tensor x = Tensor::array(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}});
        Tensor y = Tensor::array(std::vector<std::vector<float>>{{1, 2}, {4, 5}, {7, 8}, {10, 11}});

        Tensor x_train, y_train, x_test, y_test;
        Data::split(x, y, x_train, y_train, x_test, y_test, 0.5);

        REQUIRE(x_train.shape == std::vector<int>{2, 3});
        REQUIRE(y_train.shape == std::vector<int>{2, 2});
        REQUIRE(x_test.shape == std::vector<int>{2, 3});
        REQUIRE(y_test.shape == std::vector<int>{2, 2});

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                REQUIRE(x_train.at(i, j) == y_train.at(i, j));
                REQUIRE(x_test.at(i, j) == y_test.at(i, j));
            }
            REQUIRE(x_train.at(i, 2) == y_train.at(i, 1) + 1);
            REQUIRE(x_test.at(i, 2) == y_test.at(i, 1) + 1);
        }
    }
}
