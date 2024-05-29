#include <catch2/catch_all.hpp>

#include "../include/FJML/linalg.h"

using namespace FJML;

TEST_CASE("Testing linalg functions", "[linalg]") {
    SECTION("Testing dot product") {
        Tensor a = Tensor::array(std::vector<float>{1, 2, 3});
        Tensor b = Tensor::array(std::vector<float>{4, 5, 6});

        REQUIRE(LinAlg::dot_product(a, b) == 32);

        SECTION("Testing dot product with invalid shapes") {
            Tensor a = Tensor::zeros({2});
            Tensor b = Tensor::zeros({3});

            REQUIRE_THROWS_AS(LinAlg::dot_product(a, b), std::invalid_argument);
        }
    }

    SECTION("Testing matrix multiply") {
        SECTION("Testing vector times vector") {
            Tensor a = Tensor::array(std::vector<float>{1, 2, 3});
            Tensor b = Tensor::array(std::vector<float>{4, 5, 6});

            Tensor c = LinAlg::matrix_multiply(a, b);
            REQUIRE(c.shape == std::vector<int>({3, 3}));
            REQUIRE(c.at({0, 0}) == 4);
            REQUIRE(c.at({0, 1}) == 5);
            REQUIRE(c.at({0, 2}) == 6);
            REQUIRE(c.at({1, 0}) == 8);
            REQUIRE(c.at({1, 1}) == 10);
            REQUIRE(c.at({1, 2}) == 12);
            REQUIRE(c.at({2, 0}) == 12);
            REQUIRE(c.at({2, 1}) == 15);
            REQUIRE(c.at({2, 2}) == 18);
        }

        SECTION("Testing vector times matrix") {
            Tensor a = Tensor::array(std::vector<float>{1, 2, 3});
            Tensor b = Tensor::array(std::vector<std::vector<float>>{{4, 5}, {6, 7}, {8, 9}});

            Tensor c = LinAlg::matrix_multiply(a, b);
            REQUIRE(c.shape == std::vector<int>({2}));
            REQUIRE(c.at(0) == 40);
            REQUIRE(c.at(1) == 46);
        }

        SECTION("Testing matrix times vector") {
            Tensor a = Tensor::array(std::vector<std::vector<float>>{{1, 2}, {3, 4}, {5, 6}});
            Tensor b = Tensor::array(std::vector<float>{7, 8});

            Tensor c = LinAlg::matrix_multiply(a, b);
            REQUIRE(c.shape == std::vector<int>({3}));
            REQUIRE(c.at(0) == 23);
            REQUIRE(c.at(1) == 53);
            REQUIRE(c.at(2) == 83);
        }

        SECTION("Testing matrix times matrix") {
            Tensor a = Tensor::array(std::vector<std::vector<float>>{{1, 2}, {3, 4}, {5, 6}});
            Tensor b = Tensor::array(std::vector<std::vector<float>>{{7, 8, 9}, {10, 11, 12}});

            Tensor c = LinAlg::matrix_multiply(a, b);
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

        // SECTION("Testing array of matrix multiplication") {
        //     Tensor a = Tensor::array(std::vector<std::vector<std::vector<int>>>{{{1, 2}, {3, 4}, {5, 6}},
        //                                                                                   {{7, 8}, {9, 10}, {11,
        //                                                                                   12}}});
        //     Tensor b = Tensor::array(
        //         std::vector<std::vector<std::vector<int>>>{{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23,
        //         24}}});
        //
        //     Tensor c = LinAlg::matrix_multiply(a, b);
        //     REQUIRE(c.shape == std::vector<int>({2, 3, 3}));
        //     REQUIRE(c.at({0, 0, 0}) == 45);
        //     REQUIRE(c.at({0, 0, 1}) == 48);
        //     REQUIRE(c.at({0, 0, 2}) == 51);
        //     REQUIRE(c.at({0, 1, 0}) == 103);
        //     REQUIRE(c.at({0, 1, 1}) == 110);
        //     REQUIRE(c.at({0, 1, 2}) == 117);
        //     REQUIRE(c.at({0, 2, 0}) == 161);
        //     REQUIRE(c.at({0, 2, 1}) == 172);
        //     REQUIRE(c.at({0, 2, 2}) == 183);
        //     REQUIRE(c.at({1, 0, 0}) == 309);
        //     REQUIRE(c.at({1, 0, 1}) == 324);
        //     REQUIRE(c.at({1, 0, 2}) == 339);
        //     REQUIRE(c.at({1, 1, 0}) == 391);
        //     REQUIRE(c.at({1, 1, 1}) == 410);
        //     REQUIRE(c.at({1, 1, 2}) == 429);
        //     REQUIRE(c.at({1, 2, 0}) == 473);
        //     REQUIRE(c.at({1, 2, 1}) == 496);
        //     REQUIRE(c.at({1, 2, 2}) == 519);
        // }

        SECTION("Testing vector times matrix with invalid shapes") {
            Tensor a = Tensor::zeros({2});
            Tensor b = Tensor::zeros({3, 2});

            REQUIRE_THROWS_AS(LinAlg::matrix_multiply(a, b), std::invalid_argument);
        }

        SECTION("Testing matrix times vector with invalid shapes") {
            Tensor a = Tensor::zeros({2, 3});
            Tensor b = Tensor::zeros({2});

            REQUIRE_THROWS_AS(LinAlg::matrix_multiply(a, b), std::invalid_argument);
        }

        SECTION("Testing matrix times matrix with invalid shapes") {
            Tensor a = Tensor::zeros({2, 3});
            Tensor b = Tensor::zeros({4, 2});

            REQUIRE_THROWS_AS(LinAlg::matrix_multiply(a, b), std::invalid_argument);
        }

        // SECTION("Testing array of matrix multiplication with invalid shapes") {
        //     Tensor a = Tensor::zeros({2, 3, 2});
        //     Tensor b = Tensor::zeros({2, 4, 2});
        //
        //     REQUIRE_THROWS_AS(LinAlg::matrix_multiply(a, b), std::invalid_argument);
        //
        //     Tensor c = Tensor::zeros({3, 4, 5, 6, 9});
        //     Tensor d = Tensor::zeros({3, 4, 6, 9, 6});
        //
        //     REQUIRE_THROWS_AS(LinAlg::matrix_multiply(c, d), std::invalid_argument);
        // }
    }

    SECTION("Testing transpose") {
        Tensor a = Tensor::array(std::vector<std::vector<float>>{{1, 2}, {3, 4}, {5, 6}});
        Tensor b = LinAlg::transpose(a);
        REQUIRE(b.shape == std::vector<int>({2, 3}));
        REQUIRE(b.at({0, 0}) == 1);
        REQUIRE(b.at({0, 1}) == 3);
        REQUIRE(b.at({0, 2}) == 5);
        REQUIRE(b.at({1, 0}) == 2);
        REQUIRE(b.at({1, 1}) == 4);
        REQUIRE(b.at({1, 2}) == 6);

        Tensor c = Tensor::array(std::vector<float>{1, 2, 3, 4, 5});
        REQUIRE_THROWS(LinAlg::transpose(c));
    }

    SECTION("Testing sum and mean") {
        SECTION("Testing sum of vector") {
            Tensor a = Tensor::array(std::vector<float>{1, 2, 3, 4, 5});
            REQUIRE(LinAlg::sum(a) == 15);
            REQUIRE(LinAlg::mean(a) == 3);
        }

        SECTION("Testing sum of matrix") {
            Tensor a = Tensor::array(std::vector<std::vector<float>>{{1, 2}, {3, 4}, {5, 6}});
            REQUIRE(LinAlg::sum(a) == 21);
            REQUIRE(LinAlg::mean(a) == 3.5);
        }

        SECTION("Testing sum of array of matrices") {
            Tensor a = Tensor::array(std::vector<std::vector<std::vector<float>>>{{{1, 2}, {3, 4}, {5, 6}},
                                                                                  {{7, 8}, {9, 10}, {11, 12}}});
            REQUIRE(LinAlg::sum(a) == 78);
            REQUIRE(LinAlg::mean(a) == 6.5);
        }
    }

    SECTION("Testing random_choice") {
        Tensor probs = Tensor::array(std::vector<float>{0.1, 0.2, 0.3, 0.4});
        int choice = LinAlg::random_choice(probs);
        REQUIRE(choice >= 0);
        REQUIRE(choice < 4);
    }

    SECTION("Testing max") {
        Tensor a = Tensor::array(std::vector<float>{0.1, 0.2, 0.3, 0.6, 0.5, 0.4});
        REQUIRE(LinAlg::max(a) == Approx(0.6));
    }

    SECTION("Testing argmax") {
        Tensor a = Tensor::array(std::vector<float>{0.1, 0.2, 0.3, 0.6, 0.5, 0.4});
        a.reshape({2, 3});

        Tensor b = LinAlg::argmax(a, 0);
        REQUIRE(b.shape == std::vector<int>({3}));
        REQUIRE(b.at(0) == 1);
        REQUIRE(b.at(1) == 1);
        REQUIRE(b.at(2) == 1);

        Tensor c = LinAlg::argmax(a, 1);
        REQUIRE(c.shape == std::vector<int>({2}));
        REQUIRE(c.at(0) == 2);
        REQUIRE(c.at(1) == 0);

        Tensor d = LinAlg::argmax(a);
        REQUIRE(d.shape == std::vector<int>({1}));
        REQUIRE(d.at(0) == 3);
    }

    SECTION("Testing pow") {
        Tensor a = Tensor::array(std::vector<float>{1, 2, 3, 4, 5});
        Tensor b = LinAlg::pow(a, 2);
        REQUIRE(b.shape == std::vector<int>({5}));
        REQUIRE(b.at(0) == 1);
        REQUIRE(b.at(1) == 4);
        REQUIRE(b.at(2) == 9);
        REQUIRE(b.at(3) == 16);
        REQUIRE(b.at(4) == 25);

        Tensor c = Tensor::array(std::vector<std::vector<float>>{{1, 4}, {9, 16}, {25, 36}});
        Tensor d = LinAlg::pow(c, 0.5);
        REQUIRE(d.shape == std::vector<int>({3, 2}));
        REQUIRE(d.at(0, 0) == 1);
        REQUIRE(d.at(0, 1) == 2);
        REQUIRE(d.at(1, 0) == 3);
        REQUIRE(d.at(1, 1) == 4);
        REQUIRE(d.at(2, 0) == 5);
        REQUIRE(d.at(2, 1) == 6);
    }

    SECTION("Testing equal") {
        Tensor a = Tensor::array(std::vector<float>{1, 2, 3, 4, 5});
        Tensor b = Tensor::array(std::vector<float>{1, 2, 6, 4, 9});

        Tensor c = LinAlg::equal(a, b);
        REQUIRE(c.shape == std::vector<int>({5}));
        REQUIRE(c.at(0) == 1);
        REQUIRE(c.at(1) == 1);
        REQUIRE(c.at(2) == 0);
        REQUIRE(c.at(3) == 1);
        REQUIRE(c.at(4) == 0);

        Tensor d = Tensor::array(std::vector<std::vector<float>>{{1, 2}, {3, 4}, {5, 6}, {7, 8}});
        REQUIRE_THROWS(LinAlg::equal(a, d));
    }

    SECTION("Testing dense forward") {
        Tensor weights = Tensor::array(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}});
        Tensor inputs = Tensor::array(std::vector<std::vector<float>>{{1, 2}, {3, 4}});
        Tensor biases = Tensor::array(std::vector<float>{1, 2, 3});

        Tensor outputs = LinAlg::dense_forward(inputs, weights, biases);
        REQUIRE(outputs.shape == std::vector<int>({2, 3}));
        REQUIRE(outputs.at(0, 0) == 10);
        REQUIRE(outputs.at(0, 1) == 14);
        REQUIRE(outputs.at(0, 2) == 18);
        REQUIRE(outputs.at(1, 0) == 20);
        REQUIRE(outputs.at(1, 1) == 28);
        REQUIRE(outputs.at(1, 2) == 36);
    }

    SECTION("Benchmarks") {
        Tensor a{{1000}};
        Tensor b{{1000}};
        for (int i = 0; i < 1000; i++) {
            a.at(i) = i;
            b.at(i) = i;
        }

        BENCHMARK("dot product") { return FJML::LinAlg::dot_product(a, b); };

        Tensor c{{1000, 1000}};
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < 1000; j++) {
                c.at(i, j) = i + j;
            }
        }

        BENCHMARK("vector multiply matrix") { return FJML::LinAlg::matrix_multiply(a, c); };
        BENCHMARK("matrix multiply vector") { return FJML::LinAlg::matrix_multiply(c, a); };

        Tensor d{{500, 500}}, e{{500, 500}};
        for (int i = 0; i < 500; i++) {
            for (int j = 0; j < 500; j++) {
                d.at(i, j) = i + j;
                e.at(i, j) = i + j;
            }
        }

        BENCHMARK("matrix multiply matrix") { return FJML::LinAlg::matrix_multiply(d, e); };

        Tensor bias{{500}};
        for (int i = 0; i < 500; i++) {
            bias.at(i) = i;
        }
        Tensor inputs{{500, 500}};
        for (int i = 0; i < 500; i++) {
            for (int j = 0; j < 500; j++) {
                inputs.at(i, j) = i + j;
            }
        }
        BENCHMARK("dense forward") { return FJML::LinAlg::dense_forward(d, inputs, bias); };
    }
}
