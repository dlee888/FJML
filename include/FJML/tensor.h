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

#ifdef CUDA
extern cublasHandle_t handle;
extern bool handle_initialized;
#endif

/**
 * @brief An enum for what device a tensor lives on.
 */
enum Device { DEVICE_CPU, DEVICE_CUDA };

/**
 * @brief This class represents an N dimensional tensor of floats.
 * The tensor is stored as a vector, and also has a shape property.
 */
class Tensor {
  public:
    /**
     * @brief A linear array containing the data
     */
    float* data;
    /**
     * @brief The shape of the tensor
     */
    std::vector<int> shape;
    /**
     * @brief Element i contains the number of elements in the ith dimension
     */
    std::vector<int> data_size;
    /**
     * @brief The device this tensor lives on.
     */
    Device device;

    /**
     * @brief Default constructor
     */
    Tensor();

    /**
     * @brief Creates a tensor with the given shape
     * @param shape the shape of the tensor
     * @param init the initial value of the tensor, default is 0
     * @param device the device this tensor lives on
     */
    Tensor(const std::vector<int>& shape, float init = 0, Device device = DEVICE_CPU);

    /**
     * @brief Creates a tensor with the given shape
     * @param shape the shape of the tensor
     * @param device the device this tensor lives on
     */
    Tensor(const std::vector<int>& shape, Device device);

    /**
     * @brief Copy constructor
     * @param other the tensor to copy
     */
    Tensor(const Tensor& other);

    /**
     * @brief Move constructor
     * @param other the tensor to move
     */
    Tensor(Tensor&& other);

    /**
     * @brief Copy assignment operator
     * @param other the tensor to copy
     * @return a reference to this tensor
     */
    Tensor& operator=(const Tensor& other);

    /**
     * @brief Destructor
     */
    ~Tensor();

    /**
     * @brief Creates a tensor with the given shape, filled with zeros
     * @param shape the shape of the tensor
     * @param device the device this tensor lives on
     * @return a tensor with the given shape, filled with zeros
     */
    static Tensor zeros(const std::vector<int>& shape, Device device = DEVICE_CPU);

    /**
     * @brief Creates a tensor with the given shape, filled with ones
     * @param shape the shape of the tensor
     * @param device the device this tensor lives on
     * @return a tensor with the given shape, filled with ones
     */
    static Tensor ones(const std::vector<int>& shape, Device device = DEVICE_CPU);

    /**
     * @brief Creates a tensor with the given shape, filled with random values
     * @param shape the shape of the tensor
     * @param device the device this tensor lives on
     * @return a tensor with the given shape, filled with random values
     */
    static Tensor rand(const std::vector<int>& shape, Device device = DEVICE_CPU);

    /**
     * @brief Create a tensor from a given vector
     * @param vec the vector to create the tensor from
     * @param device the device this tensor lives on
     * @return a tensor with the given vector as its data
     */
    static Tensor array(const std::vector<float>& vec, Device device = DEVICE_CPU);

    /**
     * @brief Create a tensor from a given vector
     * @param vec the vector to create the tensor from
     * @param device the device this tensor lives on
     * @return a tensor with the given vector as its data
     */
    static Tensor array(const std::vector<Tensor>& vec, Device device = DEVICE_CPU);

    /**
     * @brief Create a tensor from a given vector
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
     * @brief Convert the tensor to a different device
     * @param device the device to convert to
     * @return a tensor with the same data, but on a different device
     */
    Tensor to_device(Device device) const;

    /**
     * @brief Returns the number of dimensions of the tensor
     * @return the number of dimensions of the tensor
     */
    int ndim() const;

    /**
     * @brief Returns the number of dimensions of the tensor
     * @return the number of dimensions of the tensor
     */
    int dim() const;

    /**
     * @brief Reshapes the tensor
     * @param shape the new shape of the tensor
     * @return the reshaped tensor
     */
    Tensor& reshape(const std::vector<int>& shape);

    /**
     * @brief Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    float& operator[](const std::vector<int>& index);

    /**
     * @brief Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    const float& operator[](const std::vector<int>& index) const;

    /**
     * @brief Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    float& at(const std::vector<int>& index);

    /**
     * @brief Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    const float& at(const std::vector<int>& index) const;

    /**
     * @brief Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    float& at(int index...);

    /**
     * @brief Returns the element at the given index
     * @param index the index of the element
     * @return the element at the given index
     */
    const float& at(int index...) const;

    /**
     * @brief This class is used to iterate over the elements of a tensor
     */
    class iterator {
      public:
        /**
         * @brief The tensor being iterated over
         */
        Tensor& tensor;
        /**
         * @brief The index of the current element
         */
        int index;

        /**
         * @brief Constructs an iterator
         * @param tensor the tensor to iterate over
         * @param index the index of the first element
         */
        iterator(Tensor& tensor, int index);
        /**
         * @brief Constructs an iterator
         * @param itr the iterator to copy
         */
        iterator(const iterator& itr);

        /**
         * @brief Returns the element at the current position
         * @return the element at the current position
         */
        float& operator*();

        /**
         * @brief Returns the element at the current position
         * @return the element at the current position
         */
        const float& operator*() const;

        /**
         * @brief Increments the iterator
         * @return the incremented iterator
         */
        iterator& operator++();

        /**
         * @brief Increments the iterator
         * @return the incremented iterator
         */
        iterator operator++(int);

        /**
         * @brief Decrements the iterator
         * @return the decremented iterator
         */
        iterator& operator--();

        /**
         * @brief Decrements the iterator
         * @return the decremented iterator
         */
        iterator operator--(int);

        /**
         * @brief Increments the iterator by the given amount
         * @param amount the amount to increment by
         * @return the incremented iterator
         */
        iterator& operator+=(int amount);

        /**
         * @brief Decrements the iterator by the given amount
         * @param amount the amount to decrement by
         * @return the decremented iterator
         */
        iterator& operator-=(int amount);

        /**
         * @brief Add the given amount to the iterator
         * @param amount the amount to add
         * @return the incremented iterator
         */
        iterator operator+(int amount) const;

        /**
         * @brief Subtract the given amount from the iterator
         * @param amount the amount to subtract
         * @return the decremented iterator
         */
        iterator operator-(int amount) const;

        /**
         * @brief Checks if two iterators are equal
         * @param other the other iterator
         * @return true if the iterators are equal, false otherwise
         */
        bool operator==(const iterator& other) const;

        /**
         * @brief Checks if two iterators are not equal
         * @param other the other iterator
         * @return true if the iterators are not equal, false otherwise
         */
        bool operator!=(const iterator& other) const;
    };

    /**
     * @brief Iterates over the elements of the tensor
     * @return an iterator to the first element
     */
    iterator begin();

    /**
     * @brief Iterates over the elements of the tensor
     * @return an iterator to the last element
     */
    iterator end();

    /**
     * @brief Overloads the + operator
     * @param other the other tensor
     * @return the sum of the two tensors
     */
    Tensor operator+(const Tensor& other) const;

    /**
     * @brief Overloads the += operator
     * @param other the other tensor
     * @return the sum of the two tensors
     */
    Tensor& operator+=(const Tensor& other);

    /**
     * @brief Overloads the - operator
     * @param other the other tensor
     * @return the difference of the two tensors
     */
    Tensor operator-(const Tensor& other) const;

    /**
     * @brief Overloads the -= operator
     * @param other the other tensor
     * @return the difference of the two tensors
     */
    Tensor& operator-=(const Tensor& other);

    /**
     * @brief Overloads the * operator
     *
     * Note: this is element-wise multiplication, not matrix multiplication
     *
     * @param other the other tensor
     * @return the product of the two tensors
     */
    Tensor operator*(const Tensor& other) const;

    /**
     * @brief Overloads the *= operator
     *
     * Note: this is element-wise multiplication, not matrix multiplication
     *
     * @param other the other tensor
     * @return the product of the two tensors
     */
    Tensor& operator*=(const Tensor& other);

    /**
     * @brief Overloads the / operator
     *
     * Note: this is element-wise division, not matrix division
     *
     * @param other the other tensor
     * @return the quotient of the two tensors
     */
    Tensor operator/(const Tensor& other) const;

    /**
     * @brief Overloads the /= operator
     *
     * Note: this is element-wise division, not matrix division
     *
     * @param other the other tensor
     * @return the quotient of the two tensors
     */
    Tensor& operator/=(const Tensor& other);

    /**
     * @brief Addition of a scalar to the tensor
     * @param other the scalar
     * @return the sum of the tensor and the scalar
     */
    Tensor operator+(float other) const;

    /**
     * @brief Overloads the += operator
     * @param other the scalar
     * @return the sum of the tensor and the scalar
     */
    Tensor& operator+=(float other);

    /**
     * @brief Overloads the - operator
     * @param other the scalar
     * @return the difference of the tensor and the scalar
     */
    Tensor operator-(float other) const;

    /**
     * @brief Overloads the -= operator
     * @param other the scalar
     * @return the difference of the tensor and the scalar
     */
    Tensor& operator-=(float other);

    /**
     * @brief Overloads the * operator
     * @param other the scalar
     * @return the product of the tensor and the scalar
     */
    Tensor operator*(float other) const;

    /**
     * @brief Overloads the *= operator
     * @param other the scalar
     * @return the product of the tensor and the scalar
     */
    Tensor& operator*=(float other);

    /**
     * @brief Overloads the / operator
     * @param other the scalar
     * @return the quotient of the tensor and the scalar
     */
    Tensor operator/(float other) const;

    /**
     * @brief Overloads the /= operator
     * @param other the scalar
     * @return the quotient of the tensor and the scalar
     */
    Tensor& operator/=(float other);

    /**
     * @brief Overloads the + operator
     * @param other the scalar
     * @param tensor the tensor
     * @return the sum of the tensor and the scalar
     */
    friend Tensor operator+(float other, const Tensor& tensor);

    /**
     * @brief Overloads the - operator
     * @param other the scalar
     * @param tensor the tensor
     * @return the difference of the tensor and the scalar
     */
    friend Tensor operator-(float other, const Tensor& tensor);

    /**
     * @brief Scalar times the tensor
     * @param other the scalar
     * @param tensor the tensor
     * @return the product of the tensor and the scalar
     */
    friend Tensor operator*(float other, const Tensor& tensor);

    /**
     * @brief Overloads the / operator
     * @param other the scalar
     * @param tensor the tensor
     * @return the quotient of the tensor and the scalar
     */
    friend Tensor operator/(float other, const Tensor& tensor);

    /**
     * @brief Negation of the tensor
     * @return the negation of the tensor
     */
    Tensor operator-() const;

    /**
     * @brief Overloads the << operator
     * @param os the output stream
     * @param tensor the tensor
     * @return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

  private:
    /**
     * @brief Helper method to print the tensor
     * @param os the output stream
     * @param dim the current dimension
     * @param index the current index
     */
    void print(std::ostream& os, int dim, int index) const;

  public:
    /**
     * @brief Overloads the == operator
     * @param other the other tensor
     * @return true if the two tensors are equal, false otherwise
     */
    bool operator==(const Tensor& other) const;

    /**
     * @brief Overloads the != operator
     * @param other the other tensor
     * @return true if the two tensors are not equal, false otherwise
     */
    bool operator!=(const Tensor& other) const;

    /**
     * @brief Applies a function to each element of the tensor
     *
     * Note: modifies the tensor in place
     *
     * @param f the function
     * @return the tensor with the function applied to each element
     */
    Tensor& apply_function(std::function<float(float)> f);

    /**
     * @brief Applies a function to each element of the tensor
     *
     * Note: does not modify the tensor in place
     *
     * @param f the function
     * @return the tensor with the function applied to each element
     */
    Tensor calc_function(std::function<float(float)> f) const;

    /**
     * @brief Applies a function to each element of two tensors
     *
     * Note: modifies the tensor in place
     *
     * @param f the function
     * @param other the other tensor
     * @return the tensor with the function applied to each element
     */
    Tensor& apply_function(std::function<float(float, float)> f, const Tensor& other);

    /**
     * @brief Applies a function to each element of two tensors
     *
     * Note: does not modify the tensor in place
     *
     * @param f the function
     * @param other the other tensor
     * @return the tensor with the function applied to each element
     */
    Tensor calc_function(std::function<float(float, float)> f, const Tensor& other) const;
};

} // namespace FJML

#endif
