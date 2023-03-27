#ifndef LINALG_INCLUDED
#define LINALG_INCLUDED

#include <cassert>
#include <random>

#include "tensor.h"

namespace FJML {

namespace LinAlg {

/**
 * @brief Compute the dot product of two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @return The dot product of the two vectors.
 */
inline double dotProduct(const layer_vals& a, const layer_vals& b) {
    assert(a.size() == b.size());
    double res = 0;
    for (int i = 0; i < (int)a.size(); i++) {
        res += a[i] * b[i];
    }
    return res;
}

/**
 * @brief Multiplies a vector by a matrix.
 * @param a The vector.
 * @param b The matrix.
 * @return The product of the vector and the matrix.
 */
inline layer_vals matrixMultiply(const layer_vals& a, const weights& w) {
    assert(a.size() == w.size());
    layer_vals res((int)w[0].size());
    int i, j;
#pragma omp parallel for private(i, j) shared(w, a)
    for (i = 0; i < (int)w[0].size(); i++) {
        for (j = 0; j < (int)a.size(); j++) {
            res[i] += w[j][i] * a[j];
        }
    }
    return res;
}

/**
 * @brief Multiplies a matrix by a vector.
 * @param a The matrix.
 * @param b The vector.
 * @return The product of the matrix and the vector.
 */
inline layer_vals matrixMultiply(const weights& w, const layer_vals& a) {
    assert(a.size() == w[0].size());
    layer_vals res((int)w.size());
    int i, j;
#pragma omp parallel for private(i, j) shared(w, a)
    for (i = 0; i < (int)w.size(); i++) {
        for (j = 0; j < (int)a.size(); j++) {
            res[i] += w[i][j] * a[j];
        }
    }
    return res;
}

/**
 * @brief Takes the largest value in a vector.
 * @param a The vector.
 * @return The index of the largest value in the vector.
 */
inline int argmax(const layer_vals& a) {
    int res = 0;
    for (int i = 1; i < (int)a.size(); i++) {
        if (a[i] > a[res]) {
            res = i;
        }
    }
    return res;
}

template <int N> inline double sum(const Tensor<N>& a) {
    double res = 0;
    for (int i = 0; i < a.size(); i++) {
        res += sum(a[i]);
    }
    return res;
}

template <> inline double sum(const Tensor<1>& a) {
    double res = 0;
    for (int i = 0; i < a.size(); i++) {
        res += a[i];
    }
    return res;
}

inline int random_choice(const Tensor<1>& a) {
    double rand_num = (double)rand() / RAND_MAX;
    for (int i = 0; i < a.size(); i++) {
        if (rand_num < a[i]) {
            return i;
        }
        rand_num -= a[i];
    }
    return a.size() - 1;
}

} // namespace LinAlg

} // namespace FJML

#endif
