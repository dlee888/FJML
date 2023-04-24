#include <catch2/catch_all.hpp>

#include "../src/FJML/linalg/tensor.h"

using namespace FJML;

TEST_CASE("Testing tensor", "[tensor]") {
    SECTION("Testing tensor construction") {
        Tensor<int> tensor({2, 3});
        REQUIRE(tensor.shape == std::vector<int>({2, 3}));
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
            REQUIRE(tensor.ndim() == 1);
            REQUIRE(tensor.data_size == std::vector<int>({3, 1}));
            REQUIRE(tensor.data != nullptr);

            REQUIRE(tensor.at(0) == 1);
            REQUIRE(tensor.at(1) == 2);
            REQUIRE(tensor.at(2) == 3);

            tensor = Tensor<int>::array(std::vector<std::vector<int>>{{1, 3}, {1, 2}, {3, 4}});
            REQUIRE(tensor.shape == std::vector<int>({3, 2}));
            REQUIRE(tensor.ndim() == 2);
            REQUIRE(tensor.data_size == std::vector<int>({6, 2, 1}));
            REQUIRE(tensor.data != nullptr);

            REQUIRE(tensor.at({0, 0}) == 1);
            REQUIRE(tensor.at({0, 1}) == 3);
            REQUIRE(tensor.at({1, 0}) == 1);
            REQUIRE(tensor.at({1, 1}) == 2);
            REQUIRE(tensor.at({2, 0}) == 3);
            REQUIRE(tensor.at({2, 1}) == 4);

            tensor = Tensor<int>::array(std::vector<std::vector<std::vector<int>>>{{{1, 3}, {1, 2}}, {{3, 4}, {1, 2}}});
            REQUIRE(tensor.shape == std::vector<int>({2, 2, 2}));
            REQUIRE(tensor.ndim() == 3);
            REQUIRE(tensor.data_size == std::vector<int>({8, 4, 2, 1}));
            REQUIRE(tensor.data != nullptr);

            REQUIRE(tensor.at({0, 0, 0}) == 1);
            REQUIRE(tensor.at({0, 0, 1}) == 3);
            REQUIRE(tensor.at({0, 1, 0}) == 1);
            REQUIRE(tensor.at({0, 1, 1}) == 2);
            REQUIRE(tensor.at({1, 0, 0}) == 3);
            REQUIRE(tensor.at({1, 0, 1}) == 4);
            REQUIRE(tensor.at({1, 1, 0}) == 1);
            REQUIRE(tensor.at({1, 1, 1}) == 2);
        }
    }

    SECTION("Test iterators") {
        Tensor<int> tensor = Tensor<int>::array(std::vector<std::vector<int>>{{1, 3}, {1, 2}, {3, 4}});

        SECTION("Test begin()") { REQUIRE(*tensor.begin() == 1); }

        SECTION("Test end()") { REQUIRE(*(tensor.end() - 1) == 4); }

        SECTION("Test iterator constructor and operations") {
            Tensor<int>::iterator it{tensor, 1};

            REQUIRE(*it == 3);
            it++;
            REQUIRE(*it == 1);
            ++it;
            REQUIRE(*it == 2);
            it += 2;
            REQUIRE(*it == 4);
            it -= 2;
            REQUIRE(*it == 2);
            it--;
            REQUIRE(*it == 1);
            --it;
            REQUIRE(*it == 3);

            REQUIRE(*(it + 1) == 1);
            REQUIRE(*(it - 1) == 1);

            *it = 5;
            REQUIRE(*it == 5);
            REQUIRE(tensor.at({0, 1}) == 5);

            const Tensor<int>::iterator const_it{tensor, 1};
            REQUIRE(*const_it == 5);

            REQUIRE(const_it == it);
            REQUIRE(const_it != it + 1);
        }

        SECTION("Test foreach") {
            int sum = 0;
            for (auto& x : tensor) {
                sum += x;
            }
            REQUIRE(sum == 14);
        }
    }

    SECTION("Test operators") {
        Tensor<int> tensor = Tensor<int>::array(std::vector<std::vector<int>>{{1, 3}, {1, 2}, {3, 4}});
        Tensor<int> tensor2 = Tensor<int>::array(std::vector<std::vector<int>>{{1, 3}, {1, 2}, {3, 4}});

        SECTION("Test operator==") { REQUIRE(tensor == tensor2); }

        SECTION("Test operator!=") { REQUIRE(!(tensor != tensor2)); }

        SECTION("Test operator+") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{2, 6}, {2, 4}, {6, 8}});
            REQUIRE(tensor + tensor2 == tensor3);
        }

        SECTION("Test operator-") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{0, 0}, {0, 0}, {0, 0}});
            REQUIRE(tensor - tensor2 == tensor3);
        }

        SECTION("Test operator*") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{1, 9}, {1, 4}, {9, 16}});
            REQUIRE(tensor * tensor2 == tensor3);
        }

        SECTION("Test operator/") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{1, 1}, {1, 1}, {1, 1}});
            REQUIRE(tensor / tensor2 == tensor3);
        }

        SECTION("Test operator+=") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{2, 6}, {2, 4}, {6, 8}});
            tensor += tensor2;
            REQUIRE(tensor == tensor3);
        }

        SECTION("Test operator-=") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{0, 0}, {0, 0}, {0, 0}});
            tensor -= tensor2;
            REQUIRE(tensor == tensor3);
        }

        SECTION("Test operator*=") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{1, 9}, {1, 4}, {9, 16}});
            tensor *= tensor2;
            REQUIRE(tensor == tensor3);
        }

        SECTION("Test operator/=") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{1, 1}, {1, 1}, {1, 1}});
            tensor /= tensor2;
            REQUIRE(tensor == tensor3);
        }

        SECTION("Test operator+ (scalar)") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{2, 4}, {2, 3}, {4, 5}});
            REQUIRE(tensor + 1 == tensor3);
        }

        SECTION("Test operator- (scalar)") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{0, 2}, {0, 1}, {2, 3}});
            REQUIRE(tensor - 1 == tensor3);
        }

        SECTION("Test operator* (scalar)") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{2, 6}, {2, 4}, {6, 8}});
            REQUIRE(tensor * 2 == tensor3);
        }

        SECTION("Test operator/ (scalar)") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{0, 1}, {0, 1}, {1, 2}});
            REQUIRE(tensor / 2 == tensor3);
        }

        SECTION("Test operator+= (scalar)") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{2, 4}, {2, 3}, {4, 5}});
            tensor += 1;
            REQUIRE(tensor == tensor3);
        }

        SECTION("Test operator-= (scalar)") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{0, 2}, {0, 1}, {2, 3}});
            tensor -= 1;
            REQUIRE(tensor == tensor3);
        }

        SECTION("Test operator*= (scalar)") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{2, 6}, {2, 4}, {6, 8}});
            tensor *= 2;
            REQUIRE(tensor == tensor3);
        }

        SECTION("Test operator/= (scalar)") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{0, 1}, {0, 1}, {1, 2}});
            tensor /= 2;
            REQUIRE(tensor == tensor3);
        }

        SECTION("Test operator+ (scalar)") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{2, 4}, {2, 3}, {4, 5}});
            REQUIRE(1 + tensor == tensor3);
        }

        SECTION("Test operator- (scalar)") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{2, 0}, {2, 1}, {0, -1}});
            REQUIRE(3 - tensor == tensor3);
        }

        SECTION("Test operator* (scalar)") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{2, 6}, {2, 4}, {6, 8}});
            REQUIRE(2 * tensor == tensor3);
        }

        SECTION("Test operator/ (scalar)") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{4, 1}, {4, 2}, {1, 1}});
            REQUIRE(4 / tensor == tensor3);
        }

        SECTION("Test negation") {
            Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{-1, -3}, {-1, -2}, {-3, -4}});
            REQUIRE(-tensor == tensor3);
        }
    }

    SECTION("Test tensor output") {
        Tensor<int> tensor = Tensor<int>::array(std::vector<std::vector<int>>{{1, 2}, {1, 2}, {2, 3}});
        std::stringstream ss;
        ss << tensor;
        REQUIRE(ss.str() == "[[1, 2], [1, 2], [2, 3]]");
    }

    SECTION("Test apply_fn") {
        Tensor<int> tensor = Tensor<int>::array(std::vector<std::vector<int>>{{1, 2}, {1, 2}, {2, 3}});
        Tensor<int> tensor2 = Tensor<int>::array(std::vector<std::vector<int>>{{2, 4}, {2, 4}, {4, 6}});

        REQUIRE(tensor.calc_function([](int x) { return x * 2; }) == tensor2);
        tensor.apply_function([](int x) { return x * 2; });
        REQUIRE(tensor == tensor2);

        Tensor<int> tensor3 = Tensor<int>::array(std::vector<std::vector<int>>{{1, 2}, {1, 2}, {2, 3}});
        Tensor<int> tensor4 = tensor2.calc_function([](int a, int b) { return a - b; }, tensor3);
        Tensor<int> tensor5 = Tensor<int>::array(std::vector<std::vector<int>>{{1, 2}, {1, 2}, {2, 3}});
        REQUIRE(tensor4 == tensor5);
        tensor2.apply_function([](int a, int b) { return a - b; }, tensor3);
        REQUIRE(tensor2 == tensor5);
    }

    SECTION("Benchmarking") {
        Tensor<double> t2 = Tensor<double>(std::vector<int>{1000000});
        Tensor<double> t3 = Tensor<double>(std::vector<int>{1000000});
        for (int i = 0; i < 1000000; i++) {
            t2.at(i) = i;
            t3.at(i) = i;
        }

        BENCHMARK("Tensor addition") { return t2 + t3; };
        BENCHMARK("Tensor subtraction") { return t2 - t3; };
        BENCHMARK("Tensor scalar multiplication") { return t2 * 2; };
        BENCHMARK("Tensor scalar division") { return t2 / 2; };
        BENCHMARK("Tensor hammard product") { return t2 * t3; };
        BENCHMARK("Tensor division") { return t2 / t3; };
    }
}
