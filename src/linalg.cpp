// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include "../include/FJML/linalg.h"
#include <cassert>

#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("fast-math")

static std::string print_shape(const FJML::Tensor& a) {
    std::string res = "(";
    for (int i = 0; i < a.dim(); i++) {
        res += std::to_string(a.shape[i]) + ", ";
    }
    res.pop_back();
    res.pop_back();
    res += ")";
    return res;
}

namespace FJML {

namespace LinAlg {

double dot_product(const Tensor& a, const Tensor& b) {
    if (a.data_size[0] != b.data_size[0]) {
        throw std::invalid_argument("The two vectors must have the same size.");
    }
    double res = 0;
    for (int i = 0; i < (int)a.data_size[0]; i++) {
        res += a.data[i] * b.data[i];
    }
    return res;
}

/**
 * Completes a micro kernel of matrix multiplication.
 * @param a The first matrix.
 * @param b The second matrix.
 * @param c The result matrix.
 * @param l The left bound of the multiplication.
 * @param r The right bound of the multiplication.
 * @param x The x coordinate of the upper left corner of the block.
 * @param y The y coordinate of the upper left corner of the block.
 * @param m The number of columns of the result matrix.
 * @param w The number of columns of the first matrix.
 */
static void micro_kernel(double* __restrict__ a, double* __restrict__ b, double* __restrict__ c, int l, int r, int x,
                         int y, int m, int w) {
    for (int k = l; k < r; k++) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 8; j++) {
                // assert(w == 969);
                // assert(m == 969);
                assert(i * m + x * m + j + y < 969 * m);
                assert(i * w + x * w + k < 969 * w);
                assert(k * m + j + y < w * m);
                c[i * m + x * m + j + y] += a[i * w + x * w + k] * b[k * m + j + y];
            }
        }
    }
}

/**
 * Completes a micro kernel of matrix multiplication.
 *
 * This version is used when the number of columns of the first matrix is not divisible by 8.
 * @param a The first matrix.
 * @param b The second matrix.
 * @param c The result matrix.
 * @param l The left bound of the multiplication.
 * @param r The right bound of the multiplication.
 * @param x The x coordinate of the upper left corner of the block.
 * @param y The y coordinate of the upper left corner of the block.
 * @param m The number of columns of the result matrix.
 * @param w The number of columns of the first matrix.
 */
static void var_micro_kernel1(double const* __restrict__ a, double const* __restrict__ b, double* __restrict__ c, int l,
                              int r, int x, int y, int m, int w) {
    // std::cerr << "var_micro_kernel1" << std::endl;
    // std::cerr << "l = " << l << ", r = " << r << ", x = " << x << ", y = " << y << ", m = " << m << ", w = " << w
    //           << std::endl;
    for (int k = l; k < r; k++) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < m - y; j++) {
                // std::cerr << "i = " << i << ", j = " << j << std::endl;
                // std::cerr << "a[" << i * w + x * w + k << "] = " << a[i * w + x * w + k] << std::endl;
                // std::cerr << "b[" << k * m + j + y << "] = " << b[k * m + j + y] << std::endl;
                // std::cerr << "c[" << i * m + x * m + j + y << "] = " << c[i * m + x * m + j + y] << std::endl;
                c[i * m + x * m + j + y] += a[i * w + x * w + k] * b[k * m + j + y];
            }
        }
    }
    // assert(0);
}

/**
 * Completes a micro kernel of matrix multiplication.
 *
 * This version is used when the number of rows of the first matrix is not divisible by 6.
 * @param a The first matrix.
 * @param b The second matrix.
 * @param c The result matrix.
 * @param l The left bound of the multiplication.
 * @param r The right bound of the multiplication.
 * @param x The x coordinate of the upper left corner of the block.
 * @param y The y coordinate of the upper left corner of the block.
 * @param m The number of columns of the result matrix.
 * @param w The number of columns of the first matrix.
 */
static void var_micro_kernel2(double const* __restrict__ a, double const* __restrict__ b, double* __restrict__ c, int l,
                              int r, int x, int y, int n, int m, int w) {
    // std::cerr << "var_micro_kernel2" << std::endl;
    // std::cerr << "l = " << l << ", r = " << r << ", x = " << x << ", y = " << y << ", n = " << n << ", m = " << m
    //           << ", w = " << w << std::endl;
    for (int k = l; k < r; k++) {
        for (int i = 0; i < n - x; i++) {
            for (int j = 0; j < 8; j++) {
                // std::cerr << "i = " << i << ", j = " << j << std::endl;
                // std::cerr << a << " " << b << " " << c << std::endl;
                // std::cerr << n << " " << m << " " << w << std::endl;
                // std::cerr << "a[" << i * w + x * w + k << "] = " << a[i * w + x * w + k] << std::endl;
                // std::cerr << "b[" << k * m + j + y << "] = " << b[k * m + j + y] << std::endl;
                // std::cerr << "c[" << i * m + x * m + j + y << "] = " << c[i * m + x * m + j + y] << std::endl;
                assert(i * m + x * m + j + y < n * m);
                assert(i * w + x * w + k < n * w);
                assert(k * m + j + y < w * m);
                // std::cerr << "a[" << i * w + x * w + k << "] = " << a[i * w + x * w + k] << std::endl;
                // std::cerr << "b[" << k * m + j + y << "] = " << b[k * m + j + y] << std::endl;
                // std::cerr << "c[" << i * m + x * m + j + y << "] = " << c[i * m + x * m + j + y] << std::endl;
                c[i * m + x * m + j + y] += a[i * w + x * w + k] * b[k * m + j + y];
                assert(i * m + x * m + j + y < n * m);
                assert(i * w + x * w + k < n * w);
                assert(k * m + j + y < w * m);
            }
        }
    }
    // assert(0);
}

/**
 * Completes a micro kernel of matrix multiplication.
 *
 * This version is used when the number of rows of the first matrix is not divisible by 6 and the number of columns of
 * the first matrix is not divisible by 8.
 * @param a The first matrix.
 * @param b The second matrix.
 * @param c The result matrix.
 * @param l The left bound of the multiplication.
 * @param r The right bound of the multiplication.
 * @param x The x coordinate of the upper left corner of the block.
 * @param y The y coordinate of the upper left corner of the block.
 * @param m The number of columns of the result matrix.
 * @param w The number of columns of the first matrix.
 */
static void var_micro_kernel3(double const* __restrict__ a, double const* __restrict__ b, double* __restrict__ c, int l,
                              int r, int x, int y, int n, int m, int w) {
    // std::cerr << "var_micro_kernel3" << std::endl;
    // std::cerr << "l = " << l << ", r = " << r << ", x = " << x << ", y = " << y << ", n = " << n << ", m = " << m
    //           << ", w = " << w << std::endl;
    for (int k = l; k < r; k++) {
        for (int i = 0; i < n - x; i++) {
            for (int j = 0; j < m - y; j++) {
                // std::cerr << "i = " << i << ", j = " << j << std::endl;
                // std::cerr << "a[" << i * w + x * w + k << "] = " << a[i * w + x * w + k] << std::endl;
                // std::cerr << "b[" << k * m + j + y << "] = " << b[k * m + j + y] << std::endl;
                // std::cerr << "c[" << i * m + x * m + j + y << "] = " << c[i * m + x * m + j + y] << std::endl;
                c[i * m + x * m + j + y] += a[i * w + x * w + k] * b[k * m + j + y];
            }
        }
    }
    // assert(0);
}

Tensor matrix_multiply(const Tensor& a, const Tensor& b) {
#ifdef CUDA
    if (!handle_initialized) {
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Cublas initialization failed");
        }
        handle_initialized = true;
    }
#endif
    if (a.dim() == 1 && b.dim() == 1) {
        if (a.device == DEVICE_CUDA && b.device == DEVICE_CUDA) {
#ifdef CUDA
            Tensor result({a.shape[0], b.shape[0]}, 0.0, DEVICE_CUDA);
            cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

            double *d_a, *d_b, *d_result;
            cudaHostGetDevicePointer(&d_a, a.data, 0);
            cudaHostGetDevicePointer(&d_b, b.data, 0);
            cudaHostGetDevicePointer(&d_result, result.data, 0);

            const double alpha = 1, beta = 0;
            status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, b.shape[0], a.shape[0], 1, &alpha, d_b, b.shape[0],
                                 d_a, 1, &beta, d_result, b.shape[0]);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Cublas matrix multiplication failed");
            }
            return result;
#endif
        } else {
            Tensor result({a.shape[0], b.shape[0]});
            for (int i = 0; i < a.shape[0]; i++) {
                for (int j = 0; j < b.shape[0]; j++) {
                    result.data[i * b.shape[0] + j] = a.data[i] * b.data[j];
                }
            }
            return result;
        }
    } else if (a.dim() == 1 && b.dim() == 2) {
        if (a.shape[0] != b.shape[0]) {
            throw std::invalid_argument("Invalid matrix dimensions: " + print_shape(a) + " and " + print_shape(b));
        }
#ifdef CUDA
        if (a.device == DEVICE_CUDA && b.device == DEVICE_CUDA) {
            Tensor result({b.shape[1]}, 0.0, DEVICE_CUDA);
            cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

            double *d_a, *d_b, *d_result;
            cudaHostGetDevicePointer(&d_a, a.data, 0);
            cudaHostGetDevicePointer(&d_b, b.data, 0);
            cudaHostGetDevicePointer(&d_result, result.data, 0);

            const double alpha = 1, beta = 0;
            status = cublasDgemv(handle, CUBLAS_OP_N, b.shape[1], b.shape[0], &alpha, d_b, b.shape[1], d_a, 1, &beta,
                                 d_result, 1);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Cublas matrix multiplication failed");
            }
            return result;
        }
#endif
        Tensor result({b.shape[1]});
        for (int j = 0; j < b.shape[0]; j++) {
            for (int i = 0; i < b.shape[1]; i++) {
                result.data[i] += a.data[j] * b.data[j * b.shape[1] + i];
            }
        }
        return result;
    } else if (a.dim() == 2 && b.dim() == 1) {
        if (a.shape[1] != b.shape[0]) {
            throw std::invalid_argument("Invalid matrix dimensions: " + print_shape(a) + " and " + print_shape(b));
        }
#ifdef CUDA
        if (a.device == DEVICE_CUDA && b.device == DEVICE_CUDA) {
            Tensor result({a.shape[0]}, 0.0, DEVICE_CUDA);
            cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

            double *d_a, *d_b, *d_result;
            cudaHostGetDevicePointer(&d_a, a.data, 0);
            cudaHostGetDevicePointer(&d_b, b.data, 0);
            cudaHostGetDevicePointer(&d_result, result.data, 0);

            const double alpha = 1, beta = 0;
            status = cublasDgemv(handle, CUBLAS_OP_T, a.shape[1], a.shape[0], &alpha, d_a, a.shape[1], d_b, 1, &beta,
                                 d_result, 1);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Cublas matrix multiplication failed");
            }
            return result;
        }
#endif
        Tensor result({a.shape[0]});
        for (int i = 0; i < a.shape[0]; i++) {
            for (int j = 0; j < a.shape[1]; j++) {
                result.data[i] += a.data[i * a.shape[1] + j] * b.data[j];
            }
        }
        return result;
    } else if (a.dim() != 2 || b.dim() != 2 || a.shape[1] != b.shape[0]) {
        throw std::invalid_argument("Invalid matrix dimensions: " + print_shape(a) + " and " + print_shape(b));
    }
#ifdef CUDA
    if (a.device == DEVICE_CUDA && b.device == DEVICE_CUDA) {
        Tensor result({a.shape[0], b.shape[1]}, 0.0, DEVICE_CUDA);
        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

        double *d_a, *d_b, *d_result;
        cudaHostGetDevicePointer(&d_a, a.data, 0);
        cudaHostGetDevicePointer(&d_b, b.data, 0);
        cudaHostGetDevicePointer(&d_result, result.data, 0);

        const double alpha = 1, beta = 0;
        status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, b.shape[1], a.shape[0], a.shape[1], &alpha, d_b,
                             b.shape[1], d_a, a.shape[1], &beta, d_result, b.shape[1]);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Cublas matrix multiplication failed");
        }
        return result;
    }
#endif
    Tensor result({a.shape[0], b.shape[1]});
    const int s1 = 48, s2 = 48, s3 = 24;
#pragma omp parallel for
    for (int i3 = 0; i3 < b.shape[1]; i3 += s3) {
        for (int i2 = 0; i2 < a.shape[0]; i2 += s2) {
            for (int i1 = 0; i1 < a.shape[1]; i1 += s1) {
                for (int x = i2; x < std::min(i2 + s2, a.shape[0]); x += 6) {
                    if (x + 6 > a.shape[0]) {
                        for (int y = i3; y < std::min(i3 + s3, b.shape[1]); y += 8) {
                            // std::cerr << x << " " << y << " " << a.data << " " << b.data << std::endl;
                            if (y + 8 > b.shape[1]) {
                                var_micro_kernel3(a.data, b.data, result.data, i1, std::min(i1 + s1, a.shape[1]), x, y,
                                                  a.shape[0], b.shape[1], b.shape[0]);
                            } else {
                                var_micro_kernel2(a.data, b.data, result.data, i1, std::min(i1 + s1, a.shape[1]), x, y,
                                                  a.shape[0], b.shape[1], b.shape[0]);
                            }
                            // std::cerr << x << " " << y << " " << a.data << " " << b.data << std::endl;
                        }
                    } else {
                        for (int y = i3; y < std::min(i3 + s3, b.shape[1]); y += 8) {
                            // std::cerr << x << " " << y << " " << a.data << " " << b.data << std::endl;
                            if (y + 8 > b.shape[1]) {
                                var_micro_kernel1(a.data, b.data, result.data, i1, std::min(i1 + s1, a.shape[1]), x, y,
                                                  b.shape[1], b.shape[0]);
                            } else {
                                micro_kernel(a.data, b.data, result.data, i1, std::min(i1 + s1, a.shape[1]), x, y,
                                             b.shape[1], b.shape[0]);
                            }
                            // std::cerr << x << " " << y << " " << a.data << " " << b.data << std::endl;
                        }
                    }
                }
            }
        }
    }
    // for (int i = 0; i < a.shape[0]; i++) {
    //     for (int k = 0; k < a.shape[1]; k++) {
    //         for (int j = 0; j < b.shape[1]; j++) {
    //             result.data[i * b.shape[1] + j] += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
    //         }
    //     }
    // }
    return result;
}

Tensor transpose(const Tensor& a) {
    if (a.dim() != 2) {
        throw std::invalid_argument("Argument must be a matrix");
    }
    Tensor result({a.shape[1], a.shape[0]}, a.device);
    for (int i = 0; i < a.shape[0]; i++) {
        for (int j = 0; j < a.shape[1]; j++) {
            result.data[j * a.shape[0] + i] = a.data[i * a.shape[1] + j];
        }
    }
    return result;
}

double sum(const Tensor& a) {
    double res = 0;
    for (int i = 0; i < a.data_size[0]; i++) {
        res += a.data[i];
    }
    return res;
}

int random_choice(const Tensor& a) {
    double rand_num = (double)rand() / (double)RAND_MAX;
    for (int i = 0; i < a.data_size[0]; i++) {
        if (rand_num < a.data[i]) {
            return i;
        }
        rand_num -= a.data[i];
    }
    return a.data_size[0] - 1;
}

int argmax(const Tensor& a, int axis, int index) {
    if (axis == -1) {
        axis = a.dim() - 1;
    }
    if (axis < 0 || axis >= a.dim()) {
        throw std::invalid_argument("Invalid axis");
    }
    int result = 0;
    for (int i = 1; i < a.shape[axis]; i++) {
        if (a.data[index * a.data_size[axis] + i] > a.data[index * a.data_size[axis] + result]) {
            result = i;
        }
    }
    return result;
}

double max(const Tensor& a) {
    double result = a.data[0];
    for (int i = 1; i < a.data_size[0]; i++) {
        if (a.data[i] > result) {
            result = a.data[i];
        }
    }
    return result;
}

Tensor dense_forward(const Tensor& input, const Tensor& weights, const Tensor& bias) {
    if (input.dim() != 2 || weights.dim() != 2 || bias.dim() != 1) {
        throw std::invalid_argument("Invalid dimensions for dense layer");
    }
    if (input.shape[1] != weights.shape[0] || weights.shape[1] != bias.shape[0]) {
        throw std::invalid_argument("Invalid dimensions for dense layer");
    }
#ifdef CUDA
    if (input.device == DEVICE_CUDA && weights.device == DEVICE_CUDA && bias.device == DEVICE_CUDA) {
        if (!handle) {
            cublasStatus_t status = cublasCreate(&handle);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Cublas initialization failed");
            }
        }
        Tensor result({input.shape[0], weights.shape[1]}, 0.0, DEVICE_CUDA);
        double *d_input, *d_weights, *d_result, *d_bias;
        cudaHostGetDevicePointer(&d_input, input.data, 0);
        cudaHostGetDevicePointer(&d_weights, weights.data, 0);
        cudaHostGetDevicePointer(&d_result, result.data, 0);
        cudaHostGetDevicePointer(&d_bias, bias.data, 0);

        for (int i = 0; i < input.shape[0]; i++) {
            cublasDcopy(handle, bias.shape[0], d_bias, 1, d_result + i * bias.shape[0], 1);
        }

        const double alpha = 1, beta = 1;
        cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, weights.shape[1], input.shape[0],
                                            weights.shape[0], &alpha, d_weights, weights.shape[1], d_input,
                                            input.shape[1], &beta, d_result, weights.shape[1]);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Cublas matrix multiplication failed");
        }
        return result;
    }
#endif
    Tensor result = matrix_multiply(input, weights);
#pragma omp parallel for
    for (int i = 0; i < result.shape[0]; i++) {
        for (int j = 0; j < result.shape[1]; j++) {
            result.data[i * result.shape[1] + j] += bias.data[j];
        }
    }
    return result;
}

} // namespace LinAlg

} // namespace FJML
