#ifndef TYPES_INCLUDED
#define TYPES_INCLUDED

#include <cassert>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

namespace FJML {

/**
 * This class represents an N dimensional tensor.
 * The tensor is stored as a vector, and also has a shape property.
 * @tparam N The number of dimensions.
 */
template <int N> class Tensor : public std::vector<Tensor<N - 1>> {
  public:
	/**
	 * The shape of the tensor, stored as an array with each of its dimensions.
	 */
	std::vector<int> shape;

	/**
	 * Default constructor.
	 */
	Tensor() {}

	/**
	 * Construct a tensor with the given shape.
	 * @param _shape The shape of the tensor.
	 * @param _val The value to fill the tensor with.
	 */
	Tensor(std::vector<int> _shape, double val = 0) : shape{_shape} {
		std::vector<int> next_shape{_shape.begin() + 1, _shape.end()};
		this->resize(_shape[0], Tensor<N - 1>(next_shape, val));
	}

	/**
	 * Apply a function to each element of the tensor.
	 * @param fn The function to apply.
	 * @return The result of the function.
	 */
	Tensor<N> apply_fn(std::function<double(double)> fn) {
		Tensor<N> res(this->shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i).apply_fn(fn);
		}
		return res;
	}

	/**
	 * Output the tensor to a stream.
	 * @param os The stream to output to.
	 * @param t The tensor to output.
	 * @return The stream.
	 */
	friend std::ostream& operator<<(std::ostream& o, Tensor<N> t) {
		o << "[";
		bool first = true;
		for (Tensor<N - 1> i : t) {
			if (!first) {
				o << ", ";
			}
			o << i;
			first = false;
		}
		o << "]";
		return o;
	}

	/**
	 * Adds two tensors together.
	 * @param b The tensor to add.
	 * @return The result of the addition.
	 */
	Tensor<N> operator+(const Tensor<N>& b) const {
		assert(this->size() == b.size());
		Tensor<N> res(shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) + b[i];
		}
		return res;
	}

	/**
	 * Adds a tensor to this one.
	 * @param b The tensor to add.
	 * @return The result of the addition.
	 */
	Tensor<N> operator+=(const Tensor<N>& b) {
		assert(this->size() == b.size());
		for (int i = 0; i < (int)this->size(); i++) {
			this->at(i) += b[i];
		}
		return *this;
	}

	/**
	 * Subtracts two tensors.
	 * @param b The tensor to subtract.
	 * @return The result of the subtraction.
	 */
	Tensor<N> operator-(const Tensor<N>& b) const {
		assert(this->size() == b.size());
		Tensor<N> res(this->shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) - b[i];
		}
		return res;
	}

	/**
	 * Subtracts a tensor from this one.
	 * @param b The tensor to subtract.
	 * @return The result of the subtraction.
	 */
	Tensor<N> operator-=(const Tensor<N>& b) {
		assert(this->size() == b.size());
		for (int i = 0; i < (int)this->size(); i++) {
			this->at(i) -= b[i];
		}
		return *this;
	}

	/**
	 * Multiplies a tensor by a scalar.
	 * @param b The scalar to multiply by.
	 * @return The result of the multiplication.
	 */
	Tensor<N> operator*(const double b) const {
		Tensor<N> res(this->shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) * b;
		}
		return res;
	}

	/**
	 * @see operator*(double)
	 */
	friend Tensor<N> operator*(const double b, const Tensor<N>& a) {
		Tensor<N> res(a.shape);
		for (int i = 0; i < (int)a.size(); i++) {
			res[i] = a[i] * b;
		}
		return res;
	}

	/**
	 * Multiplies a tensor by a scalar.
	 * @param b The scalar to multiply by.
	 * @return The result of the multiplication.
	 */
	Tensor<N> operator*=(const double b) {
		for (int i = 0; i < (int)this->size(); i++) {
			this->at(i) *= b;
		}
		return *this;
	}

	/**
	 * Divides a tensor by a scalar.
	 * @param b The scalar to divide by.
	 * @return The result of the division.
	 */
	Tensor<N> operator/(const double b) const {
		Tensor<N> res(this->shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) / b;
		}
		return res;
	}

	/**
	 * Divides a tensor by a scalar.
	 * @param b The scalar to divide by.
	 * @return The result of the division.
	 */
	Tensor<N> operator/=(const double b) {
		for (int i = 0; i < (int)this->size(); i++) {
			this->at(i) /= b;
		}
		return *this;
	}

	/**
	 * Multiplies two tensors together. Computes the Hadamard product.
	 *
	 * Note: This is not matrix multiplication. This is element-wise.
	 *
	 * @param b The tensor to multiply.
	 * @return The result of the multiplication.
	 */
	Tensor<N> operator*(const Tensor<N>& b) const {
		assert(this->size() == b.size());
		Tensor<N> res(this->shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) * b[i];
		}
		return res;
	}

	/**
	 * Multiplies a tensor by this one.
	 * @param b The tensor to multiply.
	 * @return The result of the multiplication.
	 * @see operator*(Tensor<N>)
	 */
	Tensor<N> operator*=(const Tensor<N>& b) {
		assert(this->size() == b.size());
		for (int i = 0; i < (int)this->size(); i++) {
			this->at(i) *= b[i];
		}
		return *this;
	}

	/**
	 * Divides two tensors. Computes the Hadamard quotient.
	 * @param b The tensor to divide by.
	 * @return The result of the division.
	 */
	Tensor<N> operator/(const Tensor<N>& b) const {
		assert(this->size() == b.size());
		Tensor<N> res(this->shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) / b[i];
		}
		return res;
	}
};

/**
 * A tensor of rank 1.
 *
 * This is not a vector of tensors but a vector of doubles.
 */
template <> class Tensor<1> : public std::vector<double> {
  public:
	/**
	 * This tensor's shape.
	 */
	std::vector<int> shape;

	/**
	 * Default constructor.
	 */
	Tensor() {}

	/**
	 * Constructs a tensor of the given shape.
	 * @param _shape The shape of the tensor.
	 * @param _val The value to initialize the tensor with.
	 */
	Tensor(std::vector<int> _shape, double _val = 0) {
		shape = _shape;
		this->clear();
		this->resize(shape[0], _val);
	}

	/**
	 * Constructs a tensor of the given shape.
	 * @param _shape The size of the tensor.
	 * @param _val The value to initialize the tensor with.
	 */
	Tensor(int _shape, double _val = 0) {
		shape = std::vector<int>{1};
		shape[0] = _shape;
		this->clear();
		this->resize(_shape, _val);
	}

	Tensor<1> apply_fn(std::function<double(double)> fn) {
		Tensor<1> res(this->shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = fn(this->at(i));
		}
		return res;
	}

	friend std::ostream& operator<<(std::ostream& o, Tensor<1> t) {
		o << "[";
		bool first = true;
		for (double i : t) {
			if (!first) {
				o << ", ";
			}
			o << i;
			first = false;
		}
		o << "]";
		return o;
	}

	Tensor<1> operator+(const Tensor<1>& b) const {
		assert(this->size() == b.size());
		Tensor<1> res(shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) + b[i];
		}
		return res;
	}

	Tensor<1> operator+=(const Tensor<1>& b) {
		assert(this->size() == b.size());
		for (int i = 0; i < (int)this->size(); i++) {
			this->at(i) += b[i];
		}
		return *this;
	}

	Tensor<1> operator-(const Tensor<1>& b) const {
		assert(this->size() == b.size());
		Tensor<1> res(shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) - b[i];
		}
		return res;
	}

	Tensor<1> operator-=(const Tensor<1>& b) {
		assert(this->size() == b.size());
		for (int i = 0; i < (int)this->size(); i++) {
			this->at(i) -= b[i];
		}
		return *this;
	}

	Tensor<1> operator*(const double b) const {
		Tensor<1> res(this->shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) * b;
		}
		return res;
	}

	friend Tensor<1> operator*(const double b, const Tensor<1>& a) {
		Tensor<1> res(a.shape);
		for (int i = 0; i < (int)a.size(); i++) {
			res[i] = a[i] * b;
		}
		return res;
	}

	Tensor<1> operator*=(const double b) {
		for (int i = 0; i < (int)this->size(); i++) {
			this->at(i) *= b;
		}
		return *this;
	}

	Tensor<1> operator/(const double b) const {
		Tensor<1> res(this->shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) / b;
		}
		return res;
	}

	Tensor<1> operator/=(const double b) {
		for (int i = 0; i < (int)this->size(); i++) {
			this->at(i) /= b;
		}
		return *this;
	}

	Tensor<1> operator*(const Tensor<1>& b) const {
		assert(this->size() == b.size());
		Tensor<1> res(shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) * b[i];
		}
		return res;
	}

	Tensor<1> operator*=(const Tensor<1>& b) {
		assert(this->size() == b.size());
		for (int i = 0; i < (int)this->size(); i++) {
			this->at(i) *= b[i];
		}
		return *this;
	}

	Tensor<1> operator/(const Tensor<1>& b) const {
		assert(this->size() == b.size());
		Tensor<1> res(shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) / b[i];
		}
		return res;
	}
};

using layer_vals = Tensor<1>;
using weights = Tensor<2>;
using bias = Tensor<1>;

} // namespace FJML

#endif
