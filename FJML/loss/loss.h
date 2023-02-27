#ifndef LOSS_INCLUDED
#define LOSS_INCLUDED

#include <cassert>
#include <functional>

#include "../linalg/tensor.h"

namespace FJML {

/**
 * @brief Namespace for loss functions
 */
namespace Loss {

const double CLIP = 1e9;

/**
 * @brief Loss function class
 *
 * The loss function class is a functor that takes two arguments, the function and its derivative.
 * The functions should take two arguments, the first is the prediction and the second is the label.
 */
class Loss {
  public:
    std::function<double(double, double)> loss_fn, grad;

    Loss() {}
    Loss(std::function<double(double, double)> f, std::function<double(double, double)> g) {
        loss_fn = f;
        grad = g;
    }

    template <int N> double calc_loss(const Tensor<N>& obs, const Tensor<N>& pred) {
        assert(obs.size() == pred.size());
        double tot = 0;
        for (int i = 0; i < (int)obs.size(); i++) {
            tot += calc_loss(obs[i], pred[i]);
        }
        return tot;
    }

    template <int N> Tensor<N> calc_grad(const Tensor<N>& obs, const Tensor<N>& pred) {
        assert(obs.size() == pred.size());
        Tensor<N> res(obs.shape);
        for (int i = 0; i < (int)obs.size(); i++) {
            res[i] = calc_grad(obs[i], pred[i]);
        }
        return res;
    }
};

template <> inline double Loss::calc_loss(const Tensor<1>& obs, const Tensor<1>& pred) {
    assert(obs.size() == pred.size());
    double tot = 0;
    for (int i = 0; i < (int)obs.size(); i++) {
        tot += loss_fn(obs[i], pred[i]);
    }
    return tot;
}

template <> inline Tensor<1> Loss::calc_grad(const Tensor<1>& obs, const Tensor<1>& pred) {
    assert(obs.size() == pred.size());
    Tensor<1> res(obs.size());
    for (int i = 0; i < (int)obs.size(); i++) {
        res[i] = std::max(std::min(grad(obs[i], pred[i]), CLIP), -CLIP);
    }
    return res;
}

extern Loss mse, huber, crossentropy;

} // namespace Loss

} // namespace FJML

#endif
