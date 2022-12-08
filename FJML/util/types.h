#ifndef TYPES_INCLUDED
#define TYPES_INCLUDED

#include <cassert>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

namespace FJML {

template <int N> class Tensor : public std::vector<Tensor<N - 1>> {
  public:
	std::vector<int> shape;

	Tensor() {}
	Tensor(std::vector<int> _shape, double val = 0) : shape{_shape} {
		std::vector<int> next_shape{_shape.begin() + 1, _shape.end()};
		this->resize(_shape[0], Tensor<N - 1>(next_shape, val));
	}

	Tensor<N> apply_fn(std::function<double(double)> fn) {
		Tensor<N> res(this->shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i).apply_fn(fn);
		}
		return res;
	}

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

	Tensor<N> operator+(const Tensor<N>& b) const {
		assert(this->size() == b.size());
		Tensor<N> res(shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) + b[i];
		}
		return res;
	}

	Tensor<N> operator+=(const Tensor<N>& b) {
		assert(this->size() == b.size());
		for (int i = 0; i < (int)this->size(); i++) {
			this->at(i) += b[i];
		}
		return *this;
	}

	Tensor<N> operator-(const Tensor<N>& b) const {
		assert(this->size() == b.size());
		Tensor<N> res(this->shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) - b[i];
		}
		return res;
	}

	Tensor<N> operator-=(const Tensor<N>& b) {
		assert(this->size() == b.size());
		for (int i = 0; i < (int)this->size(); i++) {
			this->at(i) -= b[i];
		}
		return *this;
	}

	Tensor<N> operator*(const double b) const {
		Tensor<N> res(this->shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) * b;
		}
		return res;
	}

	friend Tensor<N> operator*(const double b, const Tensor<N>& a) {
		Tensor<N> res(a.shape);
		for (int i = 0; i < (int)a.size(); i++) {
			res[i] = a[i] * b;
		}
		return res;
	}

	Tensor<N> operator*=(const double b) {
		for (int i = 0; i < (int)this->size(); i++) {
			this->at(i) *= b;
		}
		return *this;
	}

	Tensor<N> operator/(const double b) const {
		Tensor<N> res(this->shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) / b;
		}
		return res;
	}

	Tensor<N> operator/=(const double b) {
		for (int i = 0; i < (int)this->size(); i++) {
			this->at(i) /= b;
		}
		return *this;
	}

	// Hadamard product
	Tensor<N> operator*(const Tensor<N>& b) const {
		assert(this->size() == b.size());
		Tensor<N> res(this->shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) * b[i];
		}
		return res;
	}

	Tensor<N> operator*=(const Tensor<N>& b) {
		assert(this->size() == b.size());
		for (int i = 0; i < (int)this->size(); i++) {
			this->at(i) *= b[i];
		}
		return *this;
	}

	Tensor<N> operator/(const Tensor<N>& b) const {
		assert(this->size() == b.size());
		Tensor<N> res(this->shape);
		for (int i = 0; i < (int)this->size(); i++) {
			res[i] = this->at(i) / b[i];
		}
		return res;
	}
};

template <> class Tensor<1> : public std::vector<double> {
  public:
	std::vector<int> shape;

	Tensor() {}
	Tensor(std::vector<int> _shape, double _val = 0) {
		shape = _shape;
		this->clear();
		this->resize(shape[0], _val);
	}
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