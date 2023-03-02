#include <catch2/catch_all.hpp>

#include "../FJML/linalg/tensor.h"

using namespace FJML;

TEST_CASE("Testing 1d tensor", "[tensor]") {
    Tensor<1> t1 = Tensor<1>(std::vector<int>{3});
    REQUIRE(t1.size() == 3);
    REQUIRE(t1[0] == 0);
    REQUIRE(t1[1] == 0);
    REQUIRE(t1[2] == 0);
    t1[0] = 1;
    t1[1] = 2;
    t1[2] = 3;
    REQUIRE(t1[0] == 1);
    REQUIRE(t1[1] == 2);
    REQUIRE(t1[2] == 3);

    SECTION("Testing operators") {
        Tensor<1> t2 = Tensor<1>(std::vector<int>{3});
        t2[0] = 4;
        t2[1] = 5;
        t2[2] = 6;

        SECTION("Testing addition") {
            Tensor<1> t3 = t1 + t2;
            REQUIRE(t3[0] == 5);
            REQUIRE(t3[1] == 7);
            REQUIRE(t3[2] == 9);

            t1 += t2;
            REQUIRE(t1[0] == 5);
            REQUIRE(t1[1] == 7);
            REQUIRE(t1[2] == 9);
        }

        SECTION("Testing subtraction") {
            Tensor<1> t3 = t1 - t2;
            REQUIRE(t3[0] == -3);
            REQUIRE(t3[1] == -3);
            REQUIRE(t3[2] == -3);

            t1 -= t2;
            REQUIRE(t1[0] == -3);
            REQUIRE(t1[1] == -3);
            REQUIRE(t1[2] == -3);
        }

        SECTION("Testing scalar multiplication") {
            Tensor<1> t3 = t1 * 2;
            REQUIRE(t3[0] == 2);
            REQUIRE(t3[1] == 4);
            REQUIRE(t3[2] == 6);

            t1 *= 2;
            REQUIRE(t1[0] == 2);
            REQUIRE(t1[1] == 4);
            REQUIRE(t1[2] == 6);

            t3 = 2 * t1;
            REQUIRE(t3[0] == 4);
            REQUIRE(t3[1] == 8);
            REQUIRE(t3[2] == 12);
        }

        SECTION("Testing scalar division") {
            Tensor<1> t3 = t1 / 2;
            REQUIRE(t3[0] == 0.5);
            REQUIRE(t3[1] == 1);
            REQUIRE(t3[2] == 1.5);

            t1 /= 2;
            REQUIRE(t1[0] == 0.5);
            REQUIRE(t1[1] == 1);
            REQUIRE(t1[2] == 1.5);
        }

        SECTION("Testing hammard product") {
            Tensor<1> t3 = t1 * t2;
            REQUIRE(t3[0] == 4);
            REQUIRE(t3[1] == 10);
            REQUIRE(t3[2] == 18);

            t1 *= t2;
            REQUIRE(t1[0] == 4);
            REQUIRE(t1[1] == 10);
            REQUIRE(t1[2] == 18);
        }

        SECTION("Testing division") {
            Tensor<1> t3 = t1 / t2;
            REQUIRE(t3[0] == 0.25);
            REQUIRE(t3[1] == 0.4);
            REQUIRE(t3[2] == 0.5);

            t1 /= t2;
            REQUIRE(t1[0] == 0.25);
            REQUIRE(t1[1] == 0.4);
            REQUIRE(t1[2] == 0.5);
        }
    }

    SECTION("Benchmarking") {
        Tensor<1> t2 = Tensor<1>(std::vector<int>{1000000});
        Tensor<1> t3 = Tensor<1>(std::vector<int>{1000000});
        for (int i = 0; i < 1000000; i++) {
            t2[i] = i;
            t3[i] = i;
        }

        BENCHMARK("Tensor addition") { return t2 + t3; };
        BENCHMARK("Tensor subtraction") { return t2 - t3; };
        BENCHMARK("Tensor scalar multiplication") { return t2 * 2; };
        BENCHMARK("Tensor scalar division") { return t2 / 2; };
        BENCHMARK("Tensor hammard product") { return t2 * t3; };
        BENCHMARK("Tensor division") { return t2 / t3; };
    }
}

TEST_CASE("Testing 2d tensor", "[tensor]") {
    Tensor<2> t1 = Tensor<2>(std::vector<int>{3, 3});
    REQUIRE(t1.size() == 3);
    REQUIRE(t1[0][0] == 0);
    REQUIRE(t1[0][1] == 0);
    REQUIRE(t1[0][2] == 0);
    REQUIRE(t1[1][0] == 0);
    REQUIRE(t1[1][1] == 0);
    REQUIRE(t1[1][2] == 0);
    REQUIRE(t1[2][0] == 0);
    REQUIRE(t1[2][1] == 0);
    REQUIRE(t1[2][2] == 0);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            t1[i][j] = i * 3 + j;
        }
    }

    SECTION("Testing operators") {
        Tensor<2> t2 = Tensor<2>(std::vector<int>{3, 3});
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                t2[i][j] = i * 3 + j;
            }
        }

        SECTION("Testing addition") {
            Tensor<2> t3 = t1 + t2;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    REQUIRE(t3[i][j] == 2 * (i * 3 + j));
                }
            }

            t1 += t2;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    REQUIRE(t1[i][j] == 2 * (i * 3 + j));
                }
            }
        }

        SECTION("Testing subtraction") {
            Tensor<2> t3 = t1 - t2;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    REQUIRE(t3[i][j] == 0);
                }
            }

            t1 -= t2;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    REQUIRE(t1[i][j] == 0);
                }
            }
        }

        SECTION("Testing scalar multiplication") {
            Tensor<2> t3 = t1 * 2;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    REQUIRE(t3[i][j] == 2 * (i * 3 + j));
                }
            }

            t1 *= 2;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    REQUIRE(t1[i][j] == 2 * (i * 3 + j));
                }
            }
        }

        SECTION("Testing scalar division") {
            Tensor<2> t3 = t1 / 2;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    REQUIRE(t3[i][j] == 0.5 * (i * 3 + j));
                }
            }

            t1 /= 2;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    REQUIRE(t1[i][j] == 0.5 * (i * 3 + j));
                }
            }
        }

        SECTION("Testing hammard product") {
            Tensor<2> t3 = t1 * t2;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    REQUIRE(t3[i][j] == (i * 3 + j) * (i * 3 + j));
                }
            }

            t1 *= t2;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    REQUIRE(t1[i][j] == (i * 3 + j) * (i * 3 + j));
                }
            }
        }

        SECTION("Testing division") {
            t2[0][0] = 1;
            t1[0][0] = 1;
            Tensor<2> t3 = t1 / t2;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    REQUIRE(t3[i][j] == 1);
                }
            }

            t1 /= t2;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    REQUIRE(t1[i][j] == 1);
                }
            }
        }
    }

    SECTION("Benchmarks") {
        Tensor<2> t2 = Tensor<2>(std::vector<int>{1000, 1000});
        Tensor<2> t3 = Tensor<2>(std::vector<int>{1000, 1000});
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < 1000; j++) {
                t2[i][j] = i * 1000 + j;
                t3[i][j] = i * 1000 + j;
            }
        }

        BENCHMARK("Tensor addition") { return t2 + t3; };
        BENCHMARK("Tensor subtraction") { return t2 - t3; };
        BENCHMARK("Tensor scalar multiplication") { return t2 * 2; };
        BENCHMARK("Tensor scalar division") { return t2 / 2; };
        BENCHMARK("Tensor hammard product") { return t2 * t3; };
        BENCHMARK("Tensor division") { return t2 / t3; };
    }
}
