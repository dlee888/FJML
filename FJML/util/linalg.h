#pragma once

#ifndef LINALG_INCLUDED
#define LINALG_INCLUDED

#include <cassert>

#include "types.h"

namespace FJML {

namespace LinAlg {

inline double dotProduct(const layer_vals& a, const layer_vals& b) {
	assert(a.size() == b.size());
	double res = 0;
	for (int i = 0; i < (int)a.size(); i++) {
		res += a[i] * b[i];
	}
	return res;
}

inline layer_vals matrixMultiply(const layer_vals& a, const weights& w) {
	assert(a.size() == w.size());
	layer_vals res({(int)w[0].size()});
	for (int i = 0; i < (int)w[0].size(); i++) {
		for (int j = 0; j < (int)a.size(); j++) {
			res[i] += w[j][i] * a[j];
		}
	}
	return res;
}

inline layer_vals matrixMultiply(const weights& w, const layer_vals& a) {
	assert(a.size() == w[0].size());
	layer_vals res({(int)w.size()});
	for (int i = 0; i < (int)w.size(); i++) {
		for (int j = 0; j < (int)a.size(); j++) {
			res[i] += w[i][j] * a[j];
		}
	}
	return res;
}

inline int argmax(const layer_vals& a) {
	int res = 0;
	for (int i = 1; i < (int)a.size(); i++) {
		if (a[i] > a[res]) {
			res = i;
		}
	}
	return res;
}

} // namespace LinAlg

} // namespace FJML

#endif