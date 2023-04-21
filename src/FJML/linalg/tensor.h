// Copyright (c) 2022 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef TENSOR_INCLUDED
#define TENSOR_INCLUDED

#include <cassert>
#include <cstdarg>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#pragma GCC target("avx2,fma")
#pragma GCC optimize("O3,unroll-loops")

namespace FJML {

/**
 * This class represents an N dimensional tensor.
 * The tensor is stored as a vector, and also has a shape property.
 * @tparam T the type of the tensor elements
 */
template <typename T> class Tensor {
  public:
    /**
     * A linear array containing the data
     */
    std::unique_ptr<T[]> data;
    /**
     * The shape of the tensor
     */
    std::vector<int> shape;
    /**
     * Element i contains the number of elements in the ith dimension
     */
    std::vector<int> data_size;

    /**
     * Creates a tensor with the given shape
     * @param shape the shape of the tensor
     * @param init the initial value of the tensor, default is 0 or whatever the default constructor of T is
     */
    Tensor(const std::vector<int>& shape, const T& init = T()) {
        this->shape = shape;
        data_size = shape;
        for (int i = shape.size() - 2; i >= 0; i--) {
            data_size[i] *= data_size[i + 1];
        }
        data_size.push_back(1);
        data = std::unique_ptr<T[]>(new T[data_size[0]]);
        for (int i = 0; i < data_size[0]; i++) {
            data[i] = init;
        }
    }

    /**
     * Copy constructor
     * @param other the tensor to copy
     */
    Tensor(const Tensor<T>& other) {
        shape = other.shape;
        data_size = other.data_size;
        data = std::unique_ptr<T[]>(new T[data_size[0]]);
        for (int i = 0; i < data_size[0]; i++) {
            data[i] = other.data[i];
        }
    }

    /**
     * Move constructor
     * @param other the tensor to move
     */
    Tensor(Tensor<T>&& other) {
        shape = other.shape;
        data_size = other.data_size;
        data = std::move(other.data);
    }

    /**
     * Copy assignment operator
     * @param other the tensor to copy
     * @return a reference to this tensor
     */
    Tensor<T>& operator=(const Tensor<T>& other) {
        shape = other.shape;
        data_size = other.data_size;
        data = std::unique_ptr<T[]>(new T[data_size[0]]);
        for (int i = 0; i < data_size[0]; i++) {
            data[i] = other.data[i];
        }
        return *this;
    }

    /**
     * Creates a tensor with the given shape, filled with zeros (or whatever the default constructor of T is)
     * @param shape the shape of the tensor
     * @return a tensor with the given shape, filled with zeros (or whatever the default constructor of T is)
     */
    static Tensor<T> zeros(const std::vector<int>& shape) { return Tensor<T>(shape, T()); }

    /**
     * Creates a tensor with the given shape, filled with ones
     * @param shape the shape of the tensor
     * @return a tensor with the given shape, filled with ones
     */
    static Tensor<T> ones(const std::vector<int>& shape) { return Tensor<T>(shape, T(1)); }

    /**
     * Creates a tensor with the given shape, filled with random values
     * @param shape the shape of the tensor
     * @return a tensor with the given shape, filled with random values
     */
    static Tensor<T> rand(const std::vector<int>& shape) {
        Tensor<T> tensor(shape);
        for (int i = 0; i < tensor.data_size[0]; i++) {
            tensor.data[i] = T(std::rand()) / T(RAND_MAX);
        }
        return tensor;
    }

    /**
     * Create a tensor from a given vector
     * @param vec the vector to create the tensor from
     * @return a tensor with the given vector as its data
     */
    static Tensor<T> array(const std::vector<T>& vec) {
        Tensor<T> tensor({(int)vec.size()});
        for (int i = 0; i < (int)vec.size(); i++) {
            tensor.data[i] = vec[i];
        }
        return tensor;
    }

    /**
     * Create a tensor from a given vector
     * @param vec the vector to create the tensor from
     * @return a tensor with the given vector as its data
     */
    static Tensor<T> array(const std::vector<Tensor<T>>& vec) {
        std::vector<int> shape = vec[0].shape;
        shape.insert(shape.begin(), (int)vec.size());
        Tensor<T> tensor(shape);
        for (int i = 0; i < (int)vec.size(); i++) {
            for (int j = 0; j < vec[i].data_size[0]; j++) {
                tensor.data[i * vec[i].data_size[0] + j] = vec[i].data[j];
            }
        }
        return tensor;
    }

    /**
     * Create a tensor from a given vector
     * @param vec the vector to create the tensor from
     * @return a tensor with the given vector as its data
     */
    template <typename __elem> static Tensor<T> array(std::vector<__elem> vec) {
        std::vector<Tensor<T>> tensors;
        for (int i = 0; i < (int)vec.size(); i++) {
            tensors.push_back(array(vec[i]));
        }
        return array(tensors);
    }

    /**
     * Returns the number of dimensions of the tensor
     * @return the number of dimensions of the tensor
     */
    int ndim() const { return shape.size(); }

    /**
     * Returns the number of dimensions of the tensor
     * @return the number of dimensions of the tensor
     */
    int dim() const { return shape.size(); }

    /**
     * Reshapes the tensor
     * @param shape the new shape of the tensor
     */
    void reshape(const std::vector<int>& shape) {
        if (data_size[0] != std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>())) {
            throw std::invalid_argument("Cannot reshape tensor with size " + std::to_string(data_size[0]) +
                                        " to shape " + std::to_string(shape[0]));
        }
        this->shape = shape;
        data_size = shape;
        for (int i = shape.size() - 2; i >= 0; i--) {
            data_size[i] *= data_size[i + 1];
        }
        data_size.push_back(1);
    }

    /**
     * Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    T& operator[](const std::vector<int>& index) {
        assert(index.size() == shape.size());
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

    /**
     * Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    const T& operator[](const std::vector<int>& index) const {
        assert(index.size() == shape.size());
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

    /**
     * Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    T& at(const std::vector<int>& index) {
        assert(index.size() == shape.size());
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

    /**
     * Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    const T& at(const std::vector<int>& index) const {
        assert(index.size() == shape.size());
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

    /**
     * Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    T& at(int index...) {
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

    /**
     * Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    const T& at(int index...) const {
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
};

} // namespace FJML

#endif
