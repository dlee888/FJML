// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#ifndef TENSOR_INCLUDED
#define TENSOR_INCLUDED

#include <fstream>
#include <functional>
#include <vector>

#ifdef CUDA
#include <cublas_v2.h>
#endif

#pragma GCC target("avx2,fma")
#pragma GCC optimize("O3,unroll-loops")

namespace FJML {

/**
 * An enum for what device a tensor lives on.
 */
enum Device { DEVICE_CPU, DEVICE_CUDA };

/**
 * This class represents an N dimensional tensor of doubles.
 * The tensor is stored as a vector, and also has a shape property.
 */
class Tensor {
  public:
    /**
     * A linear array containing the data
     */
    double* data;
    /**
     * The shape of the tensor
     */
    std::vector<int> shape;
    /**
     * Element i contains the number of elements in the ith dimension
     */
    std::vector<int> data_size;
    /**
     * The device this tensor lives on.
     */
    const Device device;

    /**
     * Default constructor
     */
    Tensor();

    /**
     * Creates a tensor with the given shape
     * @param shape the shape of the tensor
     * @param init the initial value of the tensor, default is 0
     * @param device the device this tensor lives on
     */
    Tensor(const std::vector<int>& shape, double init = 0, Device device = DEVICE_CPU);

    /**
     * Copy constructor
     * @param other the tensor to copy
     */
    Tensor(const Tensor& other);

    /**
     * Move constructor
     * @param other the tensor to move
     */
    Tensor(Tensor&& other);

    /**
     * Copy assignment operator
     * @param other the tensor to copy
     * @return a reference to this tensor
     */
    Tensor& operator=(const Tensor& other);

    /**
     * Destructor
     */
    ~Tensor();

    /**
     * Creates a tensor with the given shape, filled with zeros
     * @param shape the shape of the tensor
     * @param device the device this tensor lives on
     * @return a tensor with the given shape, filled with zeros
     */
    static Tensor zeros(const std::vector<int>& shape, Device device = DEVICE_CPU);

    /**
     * Creates a tensor with the given shape, filled with ones
     * @param shape the shape of the tensor
     * @param device the device this tensor lives on
     * @return a tensor with the given shape, filled with ones
     */
    static Tensor ones(const std::vector<int>& shape, Device device = DEVICE_CPU);

    /**
     * Creates a tensor with the given shape, filled with random values
     * @param shape the shape of the tensor
     * @param device the device this tensor lives on
     * @return a tensor with the given shape, filled with random values
     */
    static Tensor rand(const std::vector<int>& shape, Device device = DEVICE_CPU);

    /**
     * Create a tensor from a given vector
     * @param vec the vector to create the tensor from
     * @param device the device this tensor lives on
     * @return a tensor with the given vector as its data
     */
    static Tensor array(const std::vector<double>& vec, Device device = DEVICE_CPU);

    /**
     * Create a tensor from a given vector
     * @param vec the vector to create the tensor from
     * @param device the device this tensor lives on
     * @return a tensor with the given vector as its data
     */
    static Tensor array(const std::vector<Tensor>& vec, Device device = DEVICE_CPU);

    /**
     * Create a tensor from a given vector
     * @param vec the vector to create the tensor from
     * @param device the device this tensor lives on
     * @return a tensor with the given vector as its data
     */
    template <typename __elem> static Tensor array(std::vector<__elem> vec, Device device = DEVICE_CPU) {
        std::vector<Tensor> tensors;
        for (int i = 0; i < (int)vec.size(); i++) {
            tensors.emplace_back(array(vec[i], device));
        }
        return array(tensors, device);
    }

    /**
     * Convert the tensor to a different device
     * @param device the device to convert to
     * @return a tensor with the same data, but on a different device
     */
    Tensor to_device(Device device) const;

    /**
     * Returns the number of dimensions of the tensor
     * @return the number of dimensions of the tensor
     */
    int ndim() const;

    /**
     * Returns the number of dimensions of the tensor
     * @return the number of dimensions of the tensor
     */
    int dim() const;

    /**
     * Reshapes the tensor
     * @param shape the new shape of the tensor
     */
    void reshape(const std::vector<int>& shape);

    /**
     * Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    double& operator[](const std::vector<int>& index);

    /**
     * Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    const double& operator[](const std::vector<int>& index) const;

    /**
     * Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    double& at(const std::vector<int>& index);

    /**
     * Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    const double& at(const std::vector<int>& index) const;

    /**
     * Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    double& at(int index...);

    /**
     * Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    const double& at(int index...) const;

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
        iterator(Tensor& tensor, int index);
        /**
         * Constructs an iterator
         * @param itr the iterator to copy
         */
        iterator(const iterator& itr);

        /**
         * Returns the element at the current position
         * @return the element at the current position
         */
        double& operator*();

        /**
         * Returns the element at the current position
         * @return the element at the current position
         */
        const double& operator*() const;

        /**
         * Increments the iterator
         * @return the incremented iterator
         */
        iterator& operator++();

        /**
         * Increments the iterator
         * @return the incremented iterator
         */
        iterator operator++(int);

        /**
         * Decrements the iterator
         * @return the decremented iterator
         */
        iterator& operator--();

        /**
         * Decrements the iterator
         * @return the decremented iterator
         */
        iterator operator--(int);

        /**
         * Increments the iterator by the given amount
         * @param amount the amount to increment by
         * @return the incremented iterator
         */
        iterator& operator+=(int amount);

        /**
         * Decrements the iterator by the given amount
         * @param amount the amount to decrement by
         * @return the decremented iterator
         */
        iterator& operator-=(int amount);

        /**
         * Add the given amount to the iterator
         * @param amount the amount to add
         * @return the incremented iterator
         */
        iterator operator+(int amount) const;

        /**
         * Subtract the given amount from the iterator
         * @param amount the amount to subtract
         * @return the decremented iterator
         */
        iterator operator-(int amount) const;

        /**
         * Checks if two iterators are equal
         * @param other the other iterator
         * @return true if the iterators are equal, false otherwise
         */
        bool operator==(const iterator& other) const;

        /**
         * Checks if two iterators are not equal
         * @param other the other iterator
         * @return true if the iterators are not equal, false otherwise
         */
        bool operator!=(const iterator& other) const;
    };

    /**
     * Iterates over the elements of the tensor
     * @return an iterator to the first element
     */
    iterator begin();

    /**
     * Iterates over the elements of the tensor
     * @return an iterator to the last element
     */
    iterator end();

    /**
     * Overloads the + operator
     * @param other the other tensor
     * @return the sum of the two tensors
     */
    Tensor operator+(const Tensor& other) const;

    /**
     * Overloads the += operator
     * @param other the other tensor
     * @return the sum of the two tensors
     */
    Tensor& operator+=(const Tensor& other);

    /**
     * Overloads the - operator
     * @param other the other tensor
     * @return the difference of the two tensors
     */
    Tensor operator-(const Tensor& other) const;

    /**
     * Overloads the -= operator
     * @param other the other tensor
     * @return the difference of the two tensors
     */
    Tensor& operator-=(const Tensor& other);

    /**
     * Overloads the * operator
     *
     * Note: this is element-wise multiplication, not matrix multiplication
     *
     * @param other the other tensor
     * @return the product of the two tensors
     */
    Tensor operator*(const Tensor& other) const;

    /**
     * Overloads the *= operator
     *
     * Note: this is element-wise multiplication, not matrix multiplication
     *
     * @param other the other tensor
     * @return the product of the two tensors
     */
    Tensor& operator*=(const Tensor& other);

    /**
     * Overloads the / operator
     *
     * Note: this is element-wise division, not matrix division
     *
     * @param other the other tensor
     * @return the quotient of the two tensors
     */
    Tensor operator/(const Tensor& other) const;

    /**
     * Overloads the /= operator
     *
     * Note: this is element-wise division, not matrix division
     *
     * @param other the other tensor
     * @return the quotient of the two tensors
     */
    Tensor& operator/=(const Tensor& other);

    /**
     * Addition of a scalar to the tensor
     * @param other the scalar
     * @return the sum of the tensor and the scalar
     */
    Tensor operator+(double other) const;

    /**
     * Overloads the += operator
     * @param other the scalar
     * @return the sum of the tensor and the scalar
     */
    Tensor& operator+=(double other);

    /**
     * Overloads the - operator
     * @param other the scalar
     * @return the difference of the tensor and the scalar
     */
    Tensor operator-(double other) const;

    /**
     * Overloads the -= operator
     * @param other the scalar
     * @return the difference of the tensor and the scalar
     */
    Tensor& operator-=(double other);

    /**
     * Overloads the * operator
     * @param other the scalar
     * @return the product of the tensor and the scalar
     */
    Tensor operator*(double other) const;

    /**
     * Overloads the *= operator
     * @param other the scalar
     * @return the product of the tensor and the scalar
     */
    Tensor& operator*=(double other);

    /**
     * Overloads the / operator
     * @param other the scalar
     * @return the quotient of the tensor and the scalar
     */
    Tensor operator/(double other) const;

    /**
     * Overloads the /= operator
     * @param other the scalar
     * @return the quotient of the tensor and the scalar
     */
    Tensor& operator/=(double other);

    /**
     * Overloads the + operator
     * @param other the scalar
     * @param tensor the tensor
     * @return the sum of the tensor and the scalar
     */
    friend Tensor operator+(double other, const Tensor& tensor);

    /**
     * Overloads the - operator
     * @param other the scalar
     * @param tensor the tensor
     * @return the difference of the tensor and the scalar
     */
    friend Tensor operator-(double other, const Tensor& tensor);

    /**
     * Scalar times the tensor
     * @param other the scalar
     * @param tensor the tensor
     * @return the product of the tensor and the scalar
     */
    friend Tensor operator*(double other, const Tensor& tensor);

    /**
     * Overloads the / operator
     * @param other the scalar
     * @param tensor the tensor
     * @return the quotient of the tensor and the scalar
     */
    friend Tensor operator/(double other, const Tensor& tensor);

    /**
     * Negation of the tensor
     * @return the negation of the tensor
     */
    Tensor operator-() const;

    /**
     * Overloads the << operator
     * @param os the output stream
     * @param tensor the tensor
     * @return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

  private:
    /**
     * Helper method to print the tensor
     * @param os the output stream
     * @param dim the current dimension
     * @param index the current index
     */
    void print(std::ostream& os, int dim, int index) const;

  public:
    /**
     * Overloads the == operator
     * @param other the other tensor
     * @return true if the two tensors are equal, false otherwise
     */
    bool operator==(const Tensor& other) const;

    /**
     * Overloads the != operator
     * @param other the other tensor
     * @return true if the two tensors are not equal, false otherwise
     */
    bool operator!=(const Tensor& other) const;

    /**
     * Applies a function to each element of the tensor
     *
     * Note: modifies the tensor in place
     *
     * @param f the function
     * @return the tensor with the function applied to each element
     */
    Tensor apply_function(std::function<double(double)> f) const;

    /**
     * Applies a function to each element of the tensor
     *
     * Note: does not modify the tensor in place
     *
     * @param f the function
     * @return the tensor with the function applied to each element
     */
    Tensor calc_function(std::function<double(double)> f) const;

    /**
     * Applies a function to each element of two tensors
     *
     * Note: modifies the tensor in place
     *
     * @param f the function
     * @param other the other tensor
     * @return the tensor with the function applied to each element
     */
    Tensor apply_function(std::function<double(double, double)> f, const Tensor& other) const;

    /**
     * Applies a function to each element of two tensors
     *
     * Note: does not modify the tensor in place
     *
     * @param f the function
     * @param other the other tensor
     * @return the tensor with the function applied to each element
     */
    Tensor calc_function(std::function<double(double, double)> f, const Tensor& other) const;
};

} // namespace FJML

#endif
