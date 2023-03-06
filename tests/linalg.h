#include <FJML/linalg/linalg.h>
#include <catch2/catch_all.hpp>

#include <FJML.h>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Testing linalg functions", "[linalg]") {
    SECTION("Testing dot product") {
        FJML::layer_vals a{3};
        FJML::layer_vals b{3};
        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        b[0] = 4;
        b[1] = 5;
        b[2] = 6;

        REQUIRE(FJML::LinAlg::dotProduct(a, b) == 32);
    }

    SECTION("Testing matrix multiply") {
        FJML::weights a{{3, 2}};
        FJML::layer_vals b{3};
        FJML::layer_vals c{2};
        for (int i = 0; i < 3; i++) {
            b[i] = i;
            for (int j = 0; j < 2; j++) {
                a[i][j] = i + j;
                c[j] = j;
            }
        }

        FJML::layer_vals d = FJML::LinAlg::matrixMultiply(b, a);
        REQUIRE(d.size() == 2);
        REQUIRE(d[0] == 5);
        REQUIRE(d[1] == 8);

        FJML::layer_vals e = FJML::LinAlg::matrixMultiply(a, c);
        REQUIRE(e.size() == 3);
        REQUIRE(e[0] == 1);
        REQUIRE(e[1] == 2);
        REQUIRE(e[2] == 3);
    }

    SECTION("Testing argmax") {
        FJML::layer_vals a{3};
        a[0] = 1;
        a[1] = 2;
        a[2] = 3;

        REQUIRE(FJML::LinAlg::argmax(a) == 2);
    }

    SECTION("Testing sum") {
        FJML::layer_vals a{3};
        a[0] = 1;
        a[1] = 2;
        a[2] = 3;

        REQUIRE(FJML::LinAlg::sum(a) == 6);

        FJML::Tensor<2> b{{2, 2}};
        b[0][0] = 1;
        b[0][1] = 2;
        b[1][0] = 3;
        b[1][1] = 4;

        REQUIRE(FJML::LinAlg::sum(b) == 10);
    }

    SECTION("Benchmarks") {
        FJML::layer_vals a{1000};
        FJML::layer_vals b{1000};
        for (int i = 0; i < 1000; i++) {
            a[i] = i;
            b[i] = i;
        }

        BENCHMARK("dot product") { return FJML::LinAlg::dotProduct(a, b); };

        FJML::weights c{{1000, 1000}};
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < 1000; j++) {
                c[i][j] = i + j;
            }
        }

        BENCHMARK("matrix multiply") { return FJML::LinAlg::matrixMultiply(a, c); };
    }
}

