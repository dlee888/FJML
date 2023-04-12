#include <catch2/catch_all.hpp>

#include "tensor.h"

using namespace FJML;

TEST_CASE("Testing tensor", "[tensor]") {
    SECTION("Testing tensor construction") {
        Tensor<int> tensor({2, 3});
        REQUIRE(tensor.shape == std::vector<int>({2, 3}));
        REQUIRE(tensor.size() == 6);
        REQUIRE(tensor.ndim() == 2);
        REQUIRE(tensor.data_size == std::vector<int>({6, 3, 1}));
        REQUIRE(tensor.data != nullptr);

        SECTION("Testing tensor element access") {
            tensor.at({0, 0}) = 1;
            tensor[{0, 1}] = 2;
            tensor.at(0, 2) = 3;
            tensor.at({1, 0}) = 4;
            tensor.at({1, 1}) = 5;
            tensor.at({1, 2}) = 6;

            REQUIRE(tensor.at({0, 0}) == 1);
            REQUIRE(tensor.at({0, 1}) == 2);
            REQUIRE(tensor.at({0, 2}) == 3);
            REQUIRE(tensor.at({1, 0}) == 4);
            REQUIRE(tensor[{1, 1}] == 5);
            REQUIRE(tensor.at(1, 2) == 6);

            SECTION("Testing tensor element access with invalid index") {
                REQUIRE_THROWS_AS(tensor.at({0, 3}), std::out_of_range);
                REQUIRE_THROWS_AS(tensor.at({2, 0}), std::out_of_range);
                REQUIRE_THROWS_AS(tensor.at({2, 3}), std::out_of_range);
            }

            SECTION("Testing reshape") {
                tensor.reshape({3, 2});
                REQUIRE(tensor.shape == std::vector<int>({3, 2}));
                REQUIRE(tensor.size() == 6);
                REQUIRE(tensor.ndim() == 2);
                REQUIRE(tensor.data_size == std::vector<int>({6, 2, 1}));
                REQUIRE(tensor.data != nullptr);

                REQUIRE(tensor.at({0, 0}) == 1);
                REQUIRE(tensor.at({0, 1}) == 2);
                REQUIRE(tensor.at({1, 0}) == 3);
                REQUIRE(tensor.at({1, 1}) == 4);
                REQUIRE(tensor.at({2, 0}) == 5);
                REQUIRE(tensor.at({2, 1}) == 6);

                SECTION("Testing reshape with invalid shape") {
                    REQUIRE_THROWS_AS(tensor.reshape({2, 4}), std::invalid_argument);
                }
            }
        }

        const Tensor<int> const_tensor({2, 3});
        REQUIRE(const_tensor.shape == std::vector<int>({2, 3}));
        REQUIRE(const_tensor.size() == 6);
        REQUIRE(const_tensor.ndim() == 2);
        REQUIRE(const_tensor.data_size == std::vector<int>({6, 3, 1}));
        REQUIRE(const_tensor.data != nullptr);

        SECTION("Testing const tensor element access") {
            REQUIRE(const_tensor.at({0, 0}) == 0);
            REQUIRE(const_tensor.at(0, 1) == 0);
            REQUIRE(const_tensor[{0, 2}] == 0);
            REQUIRE(const_tensor.at({1, 0}) == 0);
            REQUIRE(const_tensor.at({1, 1}) == 0);
            REQUIRE(const_tensor.at({1, 2}) == 0);

            SECTION("Testing const tensor element access with invalid index") {
                REQUIRE_THROWS_AS(const_tensor.at({0, 3}), std::out_of_range);
                REQUIRE_THROWS_AS(const_tensor.at(2, 0), std::out_of_range);
                REQUIRE_THROWS_AS(const_tensor.at({2, 3}), std::out_of_range);
            }
        }

        Tensor<int> tensor2({2, 3}, 2);
        REQUIRE(tensor2.shape == std::vector<int>({2, 3}));
        REQUIRE(tensor2.size() == 6);
        REQUIRE(tensor2.ndim() == 2);
        REQUIRE(tensor2.data_size == std::vector<int>({6, 3, 1}));
        REQUIRE(tensor2.data != nullptr);

        SECTION("Testing tensor element access with default value") {
            REQUIRE(tensor2.at({0, 0}) == 2);
            REQUIRE(tensor2.at({0, 1}) == 2);
            REQUIRE(tensor2.at({0, 2}) == 2);
            REQUIRE(tensor2.at({1, 0}) == 2);
            REQUIRE(tensor2.at({1, 1}) == 2);
            REQUIRE(tensor2.at({1, 2}) == 2);
        }

        SECTION("Testing tensor copy constructor") {
            Tensor<int> tensor3(tensor2);
            REQUIRE(tensor3.shape == std::vector<int>({2, 3}));
            REQUIRE(tensor3.size() == 6);
            REQUIRE(tensor3.ndim() == 2);
            REQUIRE(tensor3.data_size == std::vector<int>({6, 3, 1}));
            REQUIRE(tensor3.data != nullptr);

            REQUIRE(tensor3.at({0, 0}) == 2);
            REQUIRE(tensor3.at({0, 1}) == 2);
            REQUIRE(tensor3.at({0, 2}) == 2);
            REQUIRE(tensor3.at({1, 0}) == 2);
            REQUIRE(tensor3.at({1, 1}) == 2);
            REQUIRE(tensor3.at({1, 2}) == 2);
        }
    }

    SECTION("Testing static methods") {
        SECTION("Testing zeros") {
            Tensor<int> tensor = Tensor<int>::zeros({2, 3});
            REQUIRE(tensor.shape == std::vector<int>({2, 3}));
            REQUIRE(tensor.size() == 6);
            REQUIRE(tensor.ndim() == 2);
            REQUIRE(tensor.data_size == std::vector<int>({6, 3, 1}));
            REQUIRE(tensor.data != nullptr);

            REQUIRE(tensor.at({0, 0}) == 0);
            REQUIRE(tensor.at({0, 1}) == 0);
            REQUIRE(tensor.at({0, 2}) == 0);
            REQUIRE(tensor.at({1, 0}) == 0);
            REQUIRE(tensor.at({1, 1}) == 0);
            REQUIRE(tensor.at({1, 2}) == 0);
        }

        SECTION("Testing ones") {
            Tensor<int> tensor = Tensor<int>::ones({2, 3});
            REQUIRE(tensor.shape == std::vector<int>({2, 3}));
            REQUIRE(tensor.size() == 6);
            REQUIRE(tensor.ndim() == 2);
            REQUIRE(tensor.data_size == std::vector<int>({6, 3, 1}));
            REQUIRE(tensor.data != nullptr);

            REQUIRE(tensor.at({0, 0}) == 1);
            REQUIRE(tensor.at({0, 1}) == 1);
            REQUIRE(tensor.at({0, 2}) == 1);
            REQUIRE(tensor.at({1, 0}) == 1);
            REQUIRE(tensor.at({1, 1}) == 1);
            REQUIRE(tensor.at({1, 2}) == 1);
        }

        SECTION("Testing random") {
            Tensor<double> rand_tensor = Tensor<double>::rand({2, 3});

            REQUIRE(rand_tensor.shape == std::vector<int>({2, 3}));
            REQUIRE(rand_tensor.size() == 6);
            REQUIRE(rand_tensor.ndim() == 2);
            REQUIRE(rand_tensor.data_size == std::vector<int>({6, 3, 1}));
            REQUIRE(rand_tensor.data != nullptr);

            REQUIRE(rand_tensor.at({0, 0}) != 0);
            REQUIRE(rand_tensor.at({0, 1}) != 0);
            REQUIRE(rand_tensor.at({0, 2}) != 0);
            REQUIRE(rand_tensor.at({1, 0}) != 0);
            REQUIRE(rand_tensor.at({1, 1}) != 0);
            REQUIRE(rand_tensor.at({1, 2}) != 0);
        }

        SECTION("Testing array") {
            Tensor<int> tensor = Tensor<int>::array({1, 2, 3});
            REQUIRE(tensor.shape == std::vector<int>({3}));
            REQUIRE(tensor.size() == 3);
            REQUIRE(tensor.ndim() == 1);
            REQUIRE(tensor.data_size == std::vector<int>({3, 1}));
            REQUIRE(tensor.data != nullptr);

            REQUIRE(tensor.at(0) == 1);
            REQUIRE(tensor.at(1) == 2);
            REQUIRE(tensor.at(2) == 3);

            tensor = Tensor<int>::array({{1, 3}, {1, 2}, {3, 4}});
            REQUIRE(tensor.shape == std::vector<int>({3, 2}));
            REQUIRE(tensor.size() == 6);
            REQUIRE(tensor.ndim() == 2);
            REQUIRE(tensor.data_size == std::vector<int>({6, 2, 1}));
            REQUIRE(tensor.data != nullptr);

            REQUIRE(tensor.at({0, 0}) == 1);
            REQUIRE(tensor.at({0, 1}) == 3);
            REQUIRE(tensor.at({1, 0}) == 1);
            REQUIRE(tensor.at({1, 1}) == 2);
            REQUIRE(tensor.at({2, 0}) == 3);
            REQUIRE(tensor.at({2, 1}) == 4);
        }
    }
}
