#include <catch2/catch_all.hpp>

#include "../src/FJML/linalg/linalg.h"

using namespace FJML;

TEST_CASE("Testing linalg functions", "[linalg]") {
    SECTION("Testing dot product") {
        Tensor<int> a = Tensor<int>::array({1, 2, 3});
        Tensor<int> b = Tensor<int>::array({4, 5, 6});

        REQUIRE(LinAlg::dot_product(a, b) == 32);
        
        SECTION("Testing dot product with invalid shapes") {
            Tensor<int> a = Tensor<int>::zeros({2});
            Tensor<int> b = Tensor<int>::zeros({3});

            REQUIRE_THROWS_AS(LinAlg::dot_product(a, b), std::invalid_argument);
        }
    }

    SECTION("Testing matrix multiply") {
        SECTION("Testing vector times matrix") {
            Tensor<int> a = Tensor<int>::array({1, 2, 3});
            Tensor<int> b = Tensor<int>::array(std::vector<std::vector<int>>{{4, 5}, {6, 7}, {8, 9}});

            Tensor<int> c = LinAlg::matrix_multiply(a, b);
            REQUIRE(c.shape == std::vector<int>({1, 2}));
            REQUIRE(c.at({0, 0}) == 40);
            REQUIRE(c.at({0, 1}) == 46);
        }

        SECTION("Testing matrix times vector") {
            Tensor<int> a = Tensor<int>::array(std::vector<std::vector<int>>{{1, 2}, {3, 4}, {5, 6}});
            Tensor<int> b = Tensor<int>::array({7, 8});

            Tensor<int> c = LinAlg::matrix_multiply(a, b);
            REQUIRE(c.shape == std::vector<int>({3, 1}));
            REQUIRE(c.at({0, 0}) == 23);
            REQUIRE(c.at({1, 0}) == 53);
            REQUIRE(c.at({2, 0}) == 83);
        }

        SECTION("Testing matrix times matrix") {
            Tensor<int> a = Tensor<int>::array(std::vector<std::vector<int>>{{1, 2}, {3, 4}, {5, 6}});
            Tensor<int> b = Tensor<int>::array(std::vector<std::vector<int>>{{7, 8, 9}, {10, 11, 12}});

            Tensor<int> c = LinAlg::matrix_multiply(a, b);
            REQUIRE(c.shape == std::vector<int>({3, 3}));
            REQUIRE(c.at({0, 0}) == 27);
            REQUIRE(c.at({0, 1}) == 30);
            REQUIRE(c.at({0, 2}) == 33);
            REQUIRE(c.at({1, 0}) == 61);
            REQUIRE(c.at({1, 1}) == 68);
            REQUIRE(c.at({1, 2}) == 75);
            REQUIRE(c.at({2, 0}) == 95);
            REQUIRE(c.at({2, 1}) == 106);
            REQUIRE(c.at({2, 2}) == 117);
        }

        SECTION("Testing array of matrix multiplication") {
            Tensor<int> a = Tensor<int>::array(std::vector<std::vector<std::vector<int>>>{{{1, 2}, {3, 4}, {5, 6}},
                                                                                          {{7, 8}, {9, 10}, {11, 12}}});
            Tensor<int> b = Tensor<int>::array(
                std::vector<std::vector<std::vector<int>>>{{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}});

            Tensor<int> c = LinAlg::matrix_multiply(a, b);
            REQUIRE(c.shape == std::vector<int>({2, 3, 3}));
            REQUIRE(c.at({0, 0, 0}) == 45);
            REQUIRE(c.at({0, 0, 1}) == 48);
            REQUIRE(c.at({0, 0, 2}) == 51);
            REQUIRE(c.at({0, 1, 0}) == 103);
            REQUIRE(c.at({0, 1, 1}) == 110);
            REQUIRE(c.at({0, 1, 2}) == 117);
            REQUIRE(c.at({0, 2, 0}) == 161);
            REQUIRE(c.at({0, 2, 1}) == 172);
            REQUIRE(c.at({0, 2, 2}) == 183);
            REQUIRE(c.at({1, 0, 0}) == 309);
            REQUIRE(c.at({1, 0, 1}) == 324);
            REQUIRE(c.at({1, 0, 2}) == 339);
            REQUIRE(c.at({1, 1, 0}) == 391);
            REQUIRE(c.at({1, 1, 1}) == 410);
            REQUIRE(c.at({1, 1, 2}) == 429);
            REQUIRE(c.at({1, 2, 0}) == 473);
            REQUIRE(c.at({1, 2, 1}) == 496);
            REQUIRE(c.at({1, 2, 2}) == 519);
        }

        SECTION("Testing vector times matrix with invalid shapes") {
            Tensor<int> a = Tensor<int>::zeros({2});
            Tensor<int> b = Tensor<int>::zeros({3, 2});

            REQUIRE_THROWS_AS(LinAlg::matrix_multiply(a, b), std::invalid_argument);
        }

        SECTION("Testing matrix times vector with invalid shapes") {
            Tensor<int> a = Tensor<int>::zeros({2, 3});
            Tensor<int> b = Tensor<int>::zeros({2});

            REQUIRE_THROWS_AS(LinAlg::matrix_multiply(a, b), std::invalid_argument);
        }

        SECTION("Testing matrix times matrix with invalid shapes") {
            Tensor<int> a = Tensor<int>::zeros({2, 3});
            Tensor<int> b = Tensor<int>::zeros({4, 2});

            REQUIRE_THROWS_AS(LinAlg::matrix_multiply(a, b), std::invalid_argument);
        }

        SECTION("Testing array of matrix multiplication with invalid shapes") {
            Tensor<int> a = Tensor<int>::zeros({2, 3, 2});
            Tensor<int> b = Tensor<int>::zeros({2, 4, 2});

            REQUIRE_THROWS_AS(LinAlg::matrix_multiply(a, b), std::invalid_argument);

            Tensor<int> c = Tensor<int>::zeros({3, 4, 5, 6, 9});
            Tensor<int> d = Tensor<int>::zeros({3, 4, 6, 9, 6});

            REQUIRE_THROWS_AS(LinAlg::matrix_multiply(c, d), std::invalid_argument);
        }
    }

    SECTION("Testing sum") {
        SECTION("Testing sum of vector") {
            Tensor<int> a = Tensor<int>::array({1, 2, 3, 4, 5});
            REQUIRE(LinAlg::sum(a) == 15);
        }

        SECTION("Testing sum of matrix") {
            Tensor<int> a = Tensor<int>::array(std::vector<std::vector<int>>{{1, 2}, {3, 4}, {5, 6}});
            REQUIRE(LinAlg::sum(a) == 21);
        }

        SECTION("Testing sum of array of matrices") {
            Tensor<int> a = Tensor<int>::array(std::vector<std::vector<std::vector<int>>>{{{1, 2}, {3, 4}, {5, 6}},
                                                                                          {{7, 8}, {9, 10}, {11, 12}}});
            REQUIRE(LinAlg::sum(a) == 78);
        }
    }

    SECTION("Testing random_choice") {
        Tensor<double> probs = Tensor<double>::array({0.1, 0.2, 0.3, 0.4});
        int choice = LinAlg::random_choice(probs);
        REQUIRE(choice >= 0);
        REQUIRE(choice < 4);
    }
}
