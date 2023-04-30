// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef TENSOR_INCLUDED
#define TENSOR_INCLUDED

#include <cstdarg>
#include <fstream>
#include <functional>
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
     * Default constructor
     */
    Tensor() {}

    /**
     * Creates a tensor with the given shape
     * @param shape the shape of the tensor
     * @param init the initial value of the tensor, default is 0 or whatever the default constructor of T is
     */
    Tensor(const std::vector<int>& shape, const T& init = T()) {
        this->shape = shape;
        data_size = shape;
        for (int i = (int)shape.size() - 2; i >= 0; i--) {
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
        for (int i = (int)shape.size() - 2; i >= 0; i--) {
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

    /**
     * Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    const T& operator[](const std::vector<int>& index) const {
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

    /**
     * Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    T& at(const std::vector<int>& index) {
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

    /**
     * Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    const T& at(const std::vector<int>& index) const {
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

    /**
     * This class is used to iterate over the elements of a tensor
     */
    class iterator {
      public:
        /**
         * The tensor being iterated over
         */
        Tensor& tensor;
        /**
         * The index of the current element
         */
        int index;

        /**
         * Constructs an iterator
         * @param tensor the tensor to iterate over
         * @param index the index of the first element
         */
        iterator(Tensor& tensor, int index) : tensor{tensor}, index{index} {}
        /**
         * Constructs an iterator
         * @param itr the iterator to copy
         */
        iterator(const iterator& itr) : tensor{itr.tensor}, index{itr.index} {}

        /**
         * Returns the element at the current position
         * @return the element at the current position
         */
        T& operator*() { return tensor.data[index]; }

        /**
         * Returns the element at the current position
         * @return the element at the current position
         */
        const T& operator*() const { return tensor.data[index]; }

        /**
         * Increments the iterator
         * @return the incremented iterator
         */
        iterator& operator++() {
            index++;
            return *this;
        }

        /**
         * Increments the iterator
         * @return the incremented iterator
         */
        iterator operator++(int) {
            iterator itr = *this;
            index++;
            return itr;
        }

        /**
         * Decrements the iterator
         * @return the decremented iterator
         */
        iterator& operator--() {
            index--;
            return *this;
        }

        /**
         * Decrements the iterator
         * @return the decremented iterator
         */
        iterator operator--(int) {
            iterator itr = *this;
            index--;
            return itr;
        }

        /**
         * Increments the iterator by the given amount
         * @param amount the amount to increment by
         * @return the incremented iterator
         */
        iterator& operator+=(int amount) {
            index += amount;
            return *this;
        }

        /**
         * Decrements the iterator by the given amount
         * @param amount the amount to decrement by
         * @return the decremented iterator
         */
        iterator& operator-=(int amount) {
            index -= amount;
            return *this;
        }

        /**
         * Add the given amount to the iterator
         * @param amount the amount to add
         * @return the incremented iterator
         */
        iterator operator+(int amount) const {
            iterator itr = *this;
            itr += amount;
            return itr;
        }

        /**
         * Subtract the given amount from the iterator
         * @param amount the amount to subtract
         * @return the decremented iterator
         */
        iterator operator-(int amount) const {
            iterator itr = *this;
            itr -= amount;
            return itr;
        }

        /**
         * Checks if two iterators are equal
         * @param other the other iterator
         * @return true if the iterators are equal, false otherwise
         */
        bool operator==(const iterator& other) const { return index == other.index; }

        /**
         * Checks if two iterators are not equal
         * @param other the other iterator
         * @return true if the iterators are not equal, false otherwise
         */
        bool operator!=(const iterator& other) const { return index != other.index; }
    };

    /**
     * Iterates over the elements of the tensor
     * @return an iterator to the first element
     */
    iterator begin() { return iterator{*this, 0}; }

    /**
     * Iterates over the elements of the tensor
     * @return an iterator to the last element
     */
    iterator end() { return iterator{*this, data_size[0]}; }

    /**
     * Overloads the + operator
     * @param other the other tensor
     * @return the sum of the two tensors
     */
    Tensor<T> operator+(const Tensor<T>& other) const {
        if (shape != other.shape) {
            throw std::invalid_argument("Cannot add tensors with different shapes");
        }
        Tensor<T> result(shape);
        for (int i = 0; i < data_size[0]; i++) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    /**
     * Overloads the += operator
     * @param other the other tensor
     * @return the sum of the two tensors
     */
    Tensor<T>& operator+=(const Tensor<T>& other) {
        if (shape != other.shape) {
            throw std::invalid_argument("Cannot add tensors with different shapes");
        }
        for (int i = 0; i < data_size[0]; i++) {
            data[i] += other.data[i];
        }
        return *this;
    }

    /**
     * Overloads the - operator
     * @param other the other tensor
     * @return the difference of the two tensors
     */
    Tensor<T> operator-(const Tensor<T>& other) const {
        if (shape != other.shape) {
            throw std::invalid_argument("Cannot subtract tensors with different shapes");
        }
        Tensor<T> result(shape);
        for (int i = 0; i < data_size[0]; i++) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    /**
     * Overloads the -= operator
     * @param other the other tensor
     * @return the difference of the two tensors
     */
    Tensor<T>& operator-=(const Tensor<T>& other) {
        if (shape != other.shape) {
            throw std::invalid_argument("Cannot subtract tensors with different shapes");
        }
        for (int i = 0; i < data_size[0]; i++) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    /**
     * Overloads the * operator
     *
     * Note: this is element-wise multiplication, not matrix multiplication
     *
     * @param other the other tensor
     * @return the product of the two tensors
     */
    Tensor<T> operator*(const Tensor<T>& other) const {
        if (shape != other.shape) {
            throw std::invalid_argument("Cannot multiply tensors with different shapes");
        }
        Tensor<T> result(shape);
        for (int i = 0; i < data_size[0]; i++) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    /**
     * Overloads the *= operator
     *
     * Note: this is element-wise multiplication, not matrix multiplication
     *
     * @param other the other tensor
     * @return the product of the two tensors
     */
    Tensor<T>& operator*=(const Tensor<T>& other) {
        if (shape != other.shape) {
            throw std::invalid_argument("Cannot multiply tensors with different shapes");
        }
        for (int i = 0; i < data_size[0]; i++) {
            data[i] *= other.data[i];
        }
        return *this;
    }

    /**
     * Overloads the / operator
     *
     * Note: this is element-wise division, not matrix division
     *
     * @param other the other tensor
     * @return the quotient of the two tensors
     */
    Tensor<T> operator/(const Tensor<T>& other) const {
        if (shape != other.shape) {
            throw std::invalid_argument("Cannot divide tensors with different shapes");
        }
        Tensor<T> result(shape);
        for (int i = 0; i < data_size[0]; i++) {
            result.data[i] = data[i] / other.data[i];
        }
        return result;
    }

    /**
     * Overloads the /= operator
     *
     * Note: this is element-wise division, not matrix division
     *
     * @param other the other tensor
     * @return the quotient of the two tensors
     */
    Tensor<T>& operator/=(const Tensor<T>& other) {
        if (shape != other.shape) {
            throw std::invalid_argument("Cannot divide tensors with different shapes");
        }
        for (int i = 0; i < data_size[0]; i++) {
            data[i] /= other.data[i];
        }
        return *this;
    }

    /**
     * Addition of a scalar to the tensor
     * @param other the scalar
     * @return the sum of the tensor and the scalar
     */
    Tensor<T> operator+(double other) const {
        Tensor<T> result(shape);
        for (int i = 0; i < data_size[0]; i++) {
            result.data[i] = data[i] + other;
        }
        return result;
    }

    /**
     * Overloads the += operator
     * @param other the scalar
     * @return the sum of the tensor and the scalar
     */
    Tensor<T>& operator+=(double other) {
        for (int i = 0; i < data_size[0]; i++) {
            data[i] += other;
        }
        return *this;
    }

    /**
     * Overloads the - operator
     * @param other the scalar
     * @return the difference of the tensor and the scalar
     */
    Tensor<T> operator-(double other) const {
        Tensor<T> result(shape);
        for (int i = 0; i < data_size[0]; i++) {
            result.data[i] = data[i] - other;
        }
        return result;
    }

    /**
     * Overloads the -= operator
     * @param other the scalar
     * @return the difference of the tensor and the scalar
     */
    Tensor<T>& operator-=(double other) {
        for (int i = 0; i < data_size[0]; i++) {
            data[i] -= other;
        }
        return *this;
    }

    /**
     * Overloads the * operator
     * @param other the scalar
     * @return the product of the tensor and the scalar
     */
    Tensor<T> operator*(double other) const {
        Tensor<T> result(shape);
        for (int i = 0; i < data_size[0]; i++) {
            result.data[i] = data[i] * other;
        }
        return result;
    }

    /**
     * Overloads the *= operator
     * @param other the scalar
     * @return the product of the tensor and the scalar
     */
    Tensor<T>& operator*=(double other) {
        for (int i = 0; i < data_size[0]; i++) {
            data[i] *= other;
        }
        return *this;
    }

    /**
     * Overloads the / operator
     * @param other the scalar
     * @return the quotient of the tensor and the scalar
     */
    Tensor<T> operator/(double other) const {
        Tensor<T> result(shape);
        for (int i = 0; i < data_size[0]; i++) {
            result.data[i] = data[i] / other;
        }
        return result;
    }

    /**
     * Overloads the /= operator
     * @param other the scalar
     * @return the quotient of the tensor and the scalar
     */
    Tensor<T>& operator/=(double other) {
        for (int i = 0; i < data_size[0]; i++) {
            data[i] /= other;
        }
        return *this;
    }

    /**
     * Overloads the + operator
     * @param other the scalar
     * @param tensor the tensor
     * @return the sum of the tensor and the scalar
     */
    friend Tensor<T> operator+(double other, const Tensor<T>& tensor) { return tensor + other; }

    /**
     * Overloads the - operator
     * @param other the scalar
     * @param tensor the tensor
     * @return the difference of the tensor and the scalar
     */
    friend Tensor<T> operator-(double other, const Tensor<T>& tensor) {
        Tensor<T> result(tensor.shape);
        for (int i = 0; i < tensor.data_size[0]; i++) {
            result.data[i] = other - tensor.data[i];
        }
        return result;
    }

    /**
     * Scalar times the tensor
     * @param other the scalar
     * @param tensor the tensor
     * @return the product of the tensor and the scalar
     */
    friend Tensor<T> operator*(double other, const Tensor<T>& tensor) { return tensor * other; }

    /**
     * Overloads the / operator
     * @param other the scalar
     * @param tensor the tensor
     * @return the quotient of the tensor and the scalar
     */
    friend Tensor<T> operator/(double other, const Tensor<T>& tensor) {
        Tensor<T> result(tensor.shape);
        for (int i = 0; i < tensor.data_size[0]; i++) {
            result.data[i] = other / tensor.data[i];
        }
        return result;
    }

    /**
     * Negation of the tensor
     * @return the negation of the tensor
     */
    Tensor<T> operator-() const {
        Tensor<T> result(shape);
        for (int i = 0; i < data_size[0]; i++) {
            result.data[i] = -data[i];
        }
        return result;
    }

    /**
     * Overloads the << operator
     * @param os the output stream
     * @param tensor the tensor
     * @return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
        tensor.print(os, 0, 0);
        return os;
    }

  private:
    /**
     * Helper method to print the tensor
     * @param os the output stream
     * @param dim the current dimension
     * @param index the current index
     */
    void print(std::ostream& os, int dim, int index) const {
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

  public:
    /**
     * Overloads the == operator
     * @param other the other tensor
     * @return true if the two tensors are equal, false otherwise
     */
    bool operator==(const Tensor<T>& other) const {
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

    /**
     * Overloads the != operator
     * @param other the other tensor
     * @return true if the two tensors are not equal, false otherwise
     */
    bool operator!=(const Tensor<T>& other) const { return !(*this == other); }

    /**
     * Applies a function to each element of the tensor
     *
     * Note: modifies the tensor in place
     *
     * @param f the function
     * @return the tensor with the function applied to each element
     */
    Tensor<T> apply_function(std::function<T(T)> f) const {
        for (int i = 0; i < data_size[0]; i++) {
            data[i] = f(data[i]);
        }
        return *this;
    }

    /**
     * Applies a function to each element of the tensor
     *
     * Note: does not modify the tensor in place
     *
     * @param f the function
     * @return the tensor with the function applied to each element
     */
    Tensor<T> calc_function(std::function<T(T)> f) const {
        Tensor<T> result(shape);
        for (int i = 0; i < data_size[0]; i++) {
            result.data[i] = f(data[i]);
        }
        return result;
    }

    /**
     * Applies a function to each element of two tensors
     *
     * Note: modifies the tensor in place
     *
     * @param f the function
     * @param other the other tensor
     * @return the tensor with the function applied to each element
     */
    Tensor<T> apply_function(std::function<T(T, T)> f, const Tensor<T>& other) const {
        if (shape != other.shape) {
            throw std::invalid_argument("Tensors must have the same shape");
        }
        for (int i = 0; i < data_size[0]; i++) {
            data[i] = f(data[i], other.data[i]);
        }
        return *this;
    }

    /**
     * Applies a function to each element of two tensors
     *
     * Note: does not modify the tensor in place
     *
     * @param f the function
     * @param other the other tensor
     * @return the tensor with the function applied to each element
     */
    Tensor<T> calc_function(std::function<T(T, T)> f, const Tensor<T>& other) const {
        if (shape != other.shape) {
            throw std::invalid_argument("Tensors must have the same shape");
        }
        Tensor<T> result(shape);
        for (int i = 0; i < data_size[0]; i++) {
            result.data[i] = f(data[i], other.data[i]);
        }
        return result;
    }
};

} // namespace FJML

#endif
