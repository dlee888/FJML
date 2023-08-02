// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include "../include/FJML/linalg.h"

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
#pragma omp parallel for
    for (int i = 0; i < a.shape[0]; i++) {
        for (int k = 0; k < a.shape[1]; k++) {
            for (int j = 0; j < b.shape[1]; j++) {
                result.data[i * b.shape[1] + j] += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
            }
        }
    }
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

double mean(const Tensor& a) { return sum(a) / a.data_size[0]; }

Tensor pow(const Tensor& a, double b) {
    Tensor result(a.shape, a.device);
    for (int i = 0; i < a.data_size[0]; i++) {
        result.data[i] = std::pow(a.data[i], b);
    }
    return result;
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

Tensor argmax(const Tensor& a, int axis) {
    if (axis == -1) {
        axis = a.dim() - 1;
    }
    if (axis < 0 || axis >= a.dim()) {
        throw std::invalid_argument("Invalid axis");
    }
    std::vector<int> result_shape;
    for (int i = 0; i < a.dim(); i++) {
        if (i != axis) {
            result_shape.push_back(a.shape[i]);
        }
    }
    Tensor result(result_shape);
    for (int i = 0; i < result.data_size[0]; i++) {
        int max_index = 0;
        double max_value = -INFINITY;
        for (int j = 0; j < a.shape[axis]; j++) {
            int index = i % a.data_size[axis + 1] + j * a.data_size[axis + 1] +
                        i / a.data_size[axis + 1] * a.data_size[axis];
            if (a.data[index] > max_value) {
                max_value = a.data[index];
                max_index = j;
            }
        }
        result.data[i] = max_index;
    }
    return result;
}

Tensor equal(const Tensor& a, const Tensor& b) {
    if (a.data_size[0] != b.data_size[0]) {
        throw std::invalid_argument("Tensor sizes must match");
    }
    Tensor result(a.shape);
    for (int i = 0; i < a.data_size[0]; i++) {
        result.data[i] = a.data[i] == b.data[i];
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
