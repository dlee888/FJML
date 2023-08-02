#include "../include/FJML/linalg.h"

int main() {
    using namespace FJML;

    Tensor a = Tensor::ones({5000, 5000}, FJML::DEVICE_CPU);
    Tensor b = Tensor::ones({5000, 5000}, FJML::DEVICE_CPU);

    Tensor c = LinAlg::matrix_multiply(a, b);

    std::cout << c.at({0, 0}) << std::endl;

    return 0;
}
