#ifndef LOSS_INCLUDED
#define LOSS_INCLUDED

#include <cassert>
#include <functional>

#include "../util/types.h"

namespace FJML {

namespace Loss {

const double CLIP = 1e9;

class Loss {
  public:
	std::function<double(double, double)> loss_fn, grad;

	Loss() {}
	Loss(std::function<double(double, double)> f, std::function<double(double, double)> g) {
		loss_fn = f;
		grad = g;
	}

	double calc_loss(const layer_vals& obs, const layer_vals& pred) {
		assert(obs.size() == pred.size());
		double tot = 0;
		for (int i = 0; i < (int)obs.size(); i++) {
			tot += loss_fn(obs[i], pred[i]);
		}
		return tot;
	}

	layer_vals calc_grad(const layer_vals& obs, const layer_vals& pred, std::vector<bool>* mask = nullptr) {
		assert(obs.size() == pred.size());
		layer_vals res(obs.size());
		for (int i = 0; i < (int)obs.size(); i++) {
			if (mask != nullptr && !(mask->at(i))) {
				res[i] = 0;
			} else {
				res[i] = std::max(std::min(grad(obs[i], pred[i]), CLIP), -CLIP);
			}
		}
		return res;
	}
};

extern Loss mse, huber, crossentropy;

} // namespace Loss

} // namespace FJML

#endif