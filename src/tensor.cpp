// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <cstdarg>
#include <iostream>
#include <numeric>

#ifdef CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include "../include/FJML/tensor.h"

namespace FJML {

Tensor::Tensor() : data{nullptr}, shape(0), data_size{1}, device{DEVICE_CPU} {}

Tensor::Tensor(const std::vector<int>& shape, double init, Device device) : shape{shape}, device{device} {
    // std::cerr << "Creating tensor with shape: ";
    // for (int i = 0; i < (int)this->shape.size(); i++) {
    // std::cerr << this->shape[i] << " ";
    //     if (this->shape[i] <= 0) {
    //         throw std::runtime_error("Invalid shape: " + std::to_string(this->shape[i]) + " at index " +
    //         std::to_string(i));
    //     }
    // }
    // std::cerr << std::endl;
    // std::cerr << "Copying shape" << std::endl;
    data_size = shape;
    // std::cerr << "Hello there" << std::endl;
    for (int i = (int)shape.size() - 2; i >= 0; i--) {
        data_size[i] *= data_size[i + 1];
    }
    data_size.push_back(1);
    // std::cerr << "Hello there (2)" << std::endl;
    if (device == DEVICE_CPU) {
        // std::cerr << "Allocating CPU memory: " << data_size[0] << std::endl;
        data = (double*)malloc(data_size[0] * sizeof(double));
        // std::cerr << "Allocated CPU memory" << std::endl;
        for (int i = 0; i < data_size[0]; i++) {
            data[i] = init;
        }
    } else if (device == DEVICE_CUDA) {
#ifdef CUDA
        // std::cerr << "Allocating CUDA memory" << std::endl;
        cudaHostAlloc(&data, data_size[0] * sizeof(double), cudaHostAllocMapped);
        for (int i = 0; i < data_size[0]; i++) {
            data[i] = init;
        }
#else
        throw std::runtime_error("The library was not compiled with CUDA support");
#endif
    } else {
        throw std::runtime_error("Unsupported device");
    }
    // std::cerr << "Tensor created" << std::endl;
}

Tensor::Tensor(const Tensor& other) : device{other.device} {
    // std::cerr << "Copying tensor" << std::endl;
    shape = other.shape;
    data_size = other.data_size;
    // std::cerr << "Copying tensor: shape copied" << std::endl;
    if (device == DEVICE_CPU) {
        // std::cerr << "Copying tensor: CPU" << std::endl;
        if (other.data == nullptr) {
            // std::cerr << "Copying tensor: data is nullptr" << std::endl;
            data = nullptr;
            return;
        }
        data = (double*)malloc(data_size[0] * sizeof(double));
        for (int i = 0; i < data_size[0]; i++) {
            data[i] = other.data[i];
        }
    } else if (device == DEVICE_CUDA) {
#ifdef CUDA
        if (other.data == nullptr) {
            data = nullptr;
            return;
        }
        cudaHostAlloc(&data, data_size[0] * sizeof(double), cudaHostAllocMapped);
        memcpy(data, other.data, data_size[0] * sizeof(double));
#else
        throw std::runtime_error("The library was not compiled with CUDA support");
#endif
    } else {
        throw std::runtime_error("Unsupported device");
    }
}

Tensor::Tensor(Tensor&& other) : device{other.device} {
    shape = std::move(other.shape);
    data_size = std::move(other.data_size);
    data = other.data;
    other.data = nullptr;
}

Tensor::~Tensor() {
    if (device == DEVICE_CPU) {
        free(data);
    } else if (device == DEVICE_CUDA) {
#ifdef CUDA
        cudaFreeHost(data);
#else
        throw std::runtime_error("The library was not compiled with CUDA support");
#endif
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    // std::cerr << "Copying tensor operator=" << std::endl;
    shape = other.shape;
    data_size = other.data_size;
    if (device == DEVICE_CPU) {
        if (other.data == nullptr) {
            free(data);
            data = nullptr;
            return *this;
        }
        free(data);
        data = (double*)malloc(data_size[0] * sizeof(double));
        for (int i = 0; i < data_size[0]; i++) {
            data[i] = other.data[i];
        }
    } else if (device == DEVICE_CUDA) {
#ifdef CUDA
        if (other.data == nullptr) {
            cudaFreeHost(data);
            data = nullptr;
            return *this;
        }
        cudaFreeHost(data);
        cudaHostAlloc(&data, data_size[0] * sizeof(double), cudaHostAllocMapped);
        memcpy(data, other.data, data_size[0] * sizeof(double));
#else
        throw std::runtime_error("The library was not compiled with CUDA support");
#endif
    } else {
        throw std::runtime_error("Unsupported device");
    }
    return *this;
}

Tensor Tensor::zeros(const std::vector<int>& shape, Device device) { return Tensor(shape, 0.0, device); }

Tensor Tensor::ones(const std::vector<int>& shape, Device device) { return Tensor(shape, 1.0, device); }

Tensor Tensor::rand(const std::vector<int>& shape, Device device) {
    Tensor tensor(shape, 0, device);
    for (int i = 0; i < tensor.data_size[0]; i++) {
        tensor.data[i] = double(std::rand()) / double(RAND_MAX);
    }
    return tensor;
}

Tensor Tensor::array(const std::vector<double>& vec, Device device) {
    Tensor tensor({(int)vec.size()}, 0.0, device);
    for (int i = 0; i < (int)vec.size(); i++) {
        tensor.data[i] = vec[i];
    }
    return tensor;
}

Tensor Tensor::array(const std::vector<Tensor>& vec, Device device) {
    std::vector<int> shape = vec[0].shape;
    shape.insert(shape.begin(), (int)vec.size());
    Tensor tensor(shape, 0.0, device);
    for (int i = 0; i < (int)vec.size(); i++) {
        for (int j = 0; j < vec[i].data_size[0]; j++) {
            tensor.data[i * vec[i].data_size[0] + j] = vec[i].data[j];
        }
    }
    return tensor;
}

Tensor Tensor::to_device(Device device) const {
    for (int i = 0; i < (int)shape.size(); i++) {
        std::cerr << shape[i] << " ";
        if (shape[i] == 0) {
            throw std::runtime_error("Cannot move tensor to device: shape is not defined");
        }
    }
    Tensor tensor(shape, 0.0, device);
    if (device == DEVICE_CPU) {
        for (int i = 0; i < data_size[0]; i++) {
            tensor.data[i] = data[i];
        }
    } else if (device == DEVICE_CUDA) {
#ifdef CUDA
        cudaFreeHost(tensor.data);
        cudaHostAlloc(&tensor.data, data_size[0] * sizeof(double), cudaHostAllocMapped);
        memcpy(tensor.data, data, data_size[0] * sizeof(double));
#else
        throw std::runtime_error("The library was not compiled with CUDA support");
#endif
    } else {
        throw std::runtime_error("Unsupported device");
    }
    return tensor;
}

int Tensor::ndim() const { return shape.size(); }

int Tensor::dim() const { return shape.size(); }

void Tensor::reshape(const std::vector<int>& shape) {
    if (data_size[0] != std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>())) {
        throw std::invalid_argument("Cannot reshape tensor with size " + std::to_string(data_size[0]) + " to shape " +
                                    std::to_string(shape[0]));
    }
    this->shape = shape;
    data_size = shape;
    for (int i = (int)shape.size() - 2; i >= 0; i--) {
        data_size[i] *= data_size[i + 1];
    }
    data_size.push_back(1);
}

double& Tensor::operator[](const std::vector<int>& index) {
    if (index.size() != shape.size()) {
        throw std::invalid_argument("Index has " + std::to_string(index.size()) + " dimensions, but tensor has " +
                                    std::to_string(shape.size()));
    }
    int i = 0;
    for (int j = 0; j < (int)index.size(); j++) {
        if (index[j] >= shape[j]) {
            throw std::out_of_range("Index " + std::to_string(index[j]) + " is out of range for dimension " +
                                    std::to_string(j) + " with size " + std::to_string(shape[j]));
        }
        i += index[j] * data_size[j + 1];
    }
    return data[i];
}

const double& Tensor::operator[](const std::vector<int>& index) const {
    if (index.size() != shape.size()) {
        throw std::invalid_argument("Index has " + std::to_string(index.size()) + " dimensions, but tensor has " +
                                    std::to_string(shape.size()));
    }
    int i = 0;
    for (int j = 0; j < (int)index.size(); j++) {
        if (index[j] >= shape[j]) {
            throw std::out_of_range("Index " + std::to_string(index[j]) + " is out of range for dimension " +
                                    std::to_string(j) + " with size " + std::to_string(shape[j]));
        }
        i += index[j] * data_size[j + 1];
    }
    return data[i];
}

double& Tensor::at(const std::vector<int>& index) {
    if (index.size() != shape.size()) {
        throw std::invalid_argument("Index has " + std::to_string(index.size()) + " dimensions, but tensor has " +
                                    std::to_string(shape.size()));
    }
    int i = 0;
    for (int j = 0; j < (int)index.size(); j++) {
        if (index[j] >= shape[j]) {
            throw std::out_of_range("Index " + std::to_string(index[j]) + " is out of range for dimension " +
                                    std::to_string(j) + " with size " + std::to_string(shape[j]));
        }
        i += index[j] * data_size[j + 1];
    }
    return data[i];
}

const double& Tensor::at(const std::vector<int>& index) const {
    if (index.size() != shape.size()) {
        throw std::invalid_argument("Index has " + std::to_string(index.size()) + " dimensions, but tensor has " +
                                    std::to_string(shape.size()));
    }
    int i = 0;
    for (int j = 0; j < (int)index.size(); j++) {
        if (index[j] >= shape[j]) {
            throw std::out_of_range("Index " + std::to_string(index[j]) + " is out of range for dimension " +
                                    std::to_string(j) + " with size " + std::to_string(shape[j]));
        }
        i += index[j] * data_size[j + 1];
    }
    return data[i];
}

double& Tensor::at(int index...) {
    va_list args;
    va_start(args, index);

    int i = 0;
    for (int j = 0; j < (int)shape.size(); j++) {
        if (j) {
            index = va_arg(args, int);
        }
        if (index >= shape[j]) {
            throw std::out_of_range("Index " + std::to_string(index) + " is out of range for dimension " +
                                    std::to_string(j) + " with size " + std::to_string(shape[j]));
        }
        i += index * data_size[j + 1];
    }
    return data[i];
}

const double& Tensor::at(int index...) const {
    va_list args;
    va_start(args, index);

    int i = 0;
    for (int j = 0; j < (int)shape.size(); j++) {
        if (j) {
            index = va_arg(args, int);
        }
        if (index >= shape[j]) {
            throw std::out_of_range("Index " + std::to_string(index) + " is out of range for dimension " +
                                    std::to_string(j) + " with size " + std::to_string(shape[j]));
        }
        i += index * data_size[j + 1];
    }
    return data[i];
}

Tensor::iterator::iterator(Tensor& tensor, int index) : tensor{tensor}, index{index} {}

Tensor::iterator::iterator(const iterator& itr) : tensor{itr.tensor}, index{itr.index} {}

double& Tensor::iterator::operator*() { return tensor.data[index]; }

const double& Tensor::iterator::operator*() const { return tensor.data[index]; }

Tensor::iterator& Tensor::iterator::operator++() {
    index++;
    return *this;
}

Tensor::iterator Tensor::iterator::operator++(int) {
    iterator itr = *this;
    index++;
    return itr;
}

Tensor::iterator& Tensor::iterator::operator--() {
    index--;
    return *this;
}

Tensor::iterator Tensor::iterator::operator--(int) {
    iterator itr = *this;
    index--;
    return itr;
}

Tensor::iterator& Tensor::iterator::operator+=(int amount) {
    index += amount;
    return *this;
}

Tensor::iterator& Tensor::iterator::operator-=(int amount) {
    index -= amount;
    return *this;
}

Tensor::iterator Tensor::iterator::operator+(int amount) const {
    Tensor::iterator itr = *this;
    itr += amount;
    return itr;
}

Tensor::iterator Tensor::iterator::operator-(int amount) const {
    Tensor::iterator itr = *this;
    itr -= amount;
    return itr;
}

bool Tensor::iterator::operator==(const Tensor::iterator& other) const { return index == other.index; }

bool Tensor::iterator::operator!=(const Tensor::iterator& other) const { return index != other.index; }

Tensor::iterator Tensor::begin() { return Tensor::iterator{*this, 0}; }

Tensor::iterator Tensor::end() { return Tensor::iterator{*this, data_size[0]}; }

Tensor Tensor::operator+(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Cannot add tensors with different shapes");
    }
    Tensor result(shape);
    for (int i = 0; i < data_size[0]; i++) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (shape != other.shape) {
        throw std::invalid_argument("Cannot add tensors with different shapes");
    }
    for (int i = 0; i < data_size[0]; i++) {
        data[i] += other.data[i];
    }
    return *this;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Cannot subtract tensors with different shapes");
    }
    Tensor result(shape);
    for (int i = 0; i < data_size[0]; i++) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (shape != other.shape) {
        throw std::invalid_argument("Cannot subtract tensors with different shapes");
    }
    for (int i = 0; i < data_size[0]; i++) {
        data[i] -= other.data[i];
    }
    return *this;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Cannot multiply tensors with different shapes");
    }
    Tensor result(shape);
    for (int i = 0; i < data_size[0]; i++) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    if (shape != other.shape) {
        throw std::invalid_argument("Cannot multiply tensors with different shapes");
    }
    for (int i = 0; i < data_size[0]; i++) {
        data[i] *= other.data[i];
    }
    return *this;
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Cannot divide tensors with different shapes");
    }
    Tensor result(shape);
    for (int i = 0; i < data_size[0]; i++) {
        result.data[i] = data[i] / other.data[i];
    }
    return result;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    if (shape != other.shape) {
        throw std::invalid_argument("Cannot divide tensors with different shapes");
    }
    for (int i = 0; i < data_size[0]; i++) {
        data[i] /= other.data[i];
    }
    return *this;
}

Tensor Tensor::operator+(double other) const {
    Tensor result(shape);
    for (int i = 0; i < data_size[0]; i++) {
        result.data[i] = data[i] + other;
    }
    return result;
}

Tensor& Tensor::operator+=(double other) {
    for (int i = 0; i < data_size[0]; i++) {
        data[i] += other;
    }
    return *this;
}

Tensor Tensor::operator-(double other) const {
    Tensor result(shape);
    for (int i = 0; i < data_size[0]; i++) {
        result.data[i] = data[i] - other;
    }
    return result;
}

Tensor& Tensor::operator-=(double other) {
    for (int i = 0; i < data_size[0]; i++) {
        data[i] -= other;
    }
    return *this;
}

Tensor Tensor::operator*(double other) const {
    Tensor result(shape);
    for (int i = 0; i < data_size[0]; i++) {
        result.data[i] = data[i] * other;
    }
    return result;
}

Tensor& Tensor::operator*=(double other) {
    for (int i = 0; i < data_size[0]; i++) {
        data[i] *= other;
    }
    return *this;
}

Tensor Tensor::operator/(double other) const {
    Tensor result(shape);
    for (int i = 0; i < data_size[0]; i++) {
        result.data[i] = data[i] / other;
    }
    return result;
}

Tensor& Tensor::operator/=(double other) {
    for (int i = 0; i < data_size[0]; i++) {
        data[i] /= other;
    }
    return *this;
}

Tensor operator+(double other, const Tensor& tensor) { return tensor + other; }

Tensor operator-(double other, const Tensor& tensor) {
    Tensor result(tensor.shape);
    for (int i = 0; i < tensor.data_size[0]; i++) {
        result.data[i] = other - tensor.data[i];
    }
    return result;
}

Tensor operator*(double other, const Tensor& tensor) { return tensor * other; }

Tensor operator/(double other, const Tensor& tensor) {
    Tensor result(tensor.shape);
    for (int i = 0; i < tensor.data_size[0]; i++) {
        result.data[i] = other / tensor.data[i];
    }
    return result;
}

Tensor Tensor::operator-() const {
    Tensor result(shape);
    for (int i = 0; i < data_size[0]; i++) {
        result.data[i] = -data[i];
    }
    return result;
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    tensor.print(os, 0, 0);
    return os;
}

void Tensor::print(std::ostream& os, int dim, int index) const {
    if (dim == (int)shape.size() - 1) {
        os << "[";
        for (int i = 0; i < shape[dim]; i++) {
            os << data[index + i];
            if (i != shape[dim] - 1) {
                os << ", ";
            }
        }
        os << "]";
    } else {
        os << "[";
        for (int i = 0; i < shape[dim]; i++) {
            print(os, dim + 1, index + i * data_size[dim + 1]);
            if (i != shape[dim] - 1) {
                os << ", ";
            }
        }
        os << "]";
    }
}

bool Tensor::operator==(const Tensor& other) const {
    if (shape != other.shape) {
        return false;
    }
    for (int i = 0; i < data_size[0]; i++) {
        if (data[i] != other.data[i]) {
            return false;
        }
    }
    return true;
}

bool Tensor::operator!=(const Tensor& other) const { return !(*this == other); }

Tensor Tensor::apply_function(std::function<double(double)> f) const {
    for (int i = 0; i < data_size[0]; i++) {
        data[i] = f(data[i]);
    }
    return *this;
}

Tensor Tensor::calc_function(std::function<double(double)> f) const {
    Tensor result(shape);
    for (int i = 0; i < data_size[0]; i++) {
        result.data[i] = f(data[i]);
    }
    return result;
}

Tensor Tensor::apply_function(std::function<double(double, double)> f, const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Tensors must have the same shape");
    }
    for (int i = 0; i < data_size[0]; i++) {
        data[i] = f(data[i], other.data[i]);
    }
    return *this;
}

Tensor Tensor::calc_function(std::function<double(double, double)> f, const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Tensors must have the same shape");
    }
    Tensor result(shape);
    for (int i = 0; i < data_size[0]; i++) {
        result.data[i] = f(data[i], other.data[i]);
    }
    return result;
}

} // namespace FJML
