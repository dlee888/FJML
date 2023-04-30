// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "../../../include/FJML/mlp.h"

// TODO: Refactor everything

static void progress_bar(int curr, int tot, int bar_width = 69, double time_elapsed = -1) {
    float progress = (float)curr / tot;
    std::cout << "[";
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; i++) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %";
    if (time_elapsed > 0) {
        std::cout << std::fixed << std::setprecision(3) << " Time: " << time_elapsed
                  << ", ETA = " << time_elapsed * (1 - progress) / progress;
    }
    std::cout << "\r";
    std::cout.flush();
}

namespace FJML {

void MLP::grad_descent(const std::vector<Tensor<double>>& x_train, const std::vector<Tensor<double>>& y_train) {
    assert(x_train.size() == y_train.size());
    int num_inputs = x_train.size(), num_layers = layers.size();

    std::vector<std::vector<Tensor<double>>> run_res(num_layers + 1);
    run_res[0] = x_train;
    for (int i = 0; i < num_layers; i++) {
        run_res[i + 1] = layers[i]->apply(run_res[i]);
    }

    std::vector<Tensor<double>> out_grad(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        out_grad[i] = loss_fn.calc_derivative(y_train[i], run_res[num_layers][i]);
    }

    for (int i = num_layers - 1; i >= 0; i--) {
        out_grad = layers[i]->backward(run_res[i], run_res[i + 1], out_grad);
    }
}

void MLP::backwards_pass(const std::vector<Tensor<double>>& input, const std::vector<Tensor<double>>& grads) {
    int num_layers = layers.size();
    std::vector<std::vector<Tensor<double>>> run_res(num_layers + 1);
    run_res[0] = input;
    for (int i = 0; i < num_layers; i++) {
        run_res[i + 1] = layers[i]->apply(run_res[i]);
    }

    std::vector<Tensor<double>> out_grad = grads;

    for (int i = num_layers - 1; i >= 0; i--) {
        out_grad = layers[i]->backward(run_res[i], run_res[i + 1], out_grad);
    }
}

Tensor<double> MLP::run(const Tensor<double>& input) const {
    Tensor<double> result = input;
    for (Layers::Layer* l : layers) {
        result = l->apply(result);
    }
    return result;
}

double MLP::calc_loss(const std::vector<Tensor<double>>& x_test, const std::vector<Tensor<double>>& y_test) const {
    assert(x_test.size() == y_test.size());
    double total = 0;
    for (int i = 0; i < (int)x_test.size(); i++) {
        Tensor<double> y_pred = run(x_test[i]);
        total += loss_fn.calc_loss(y_test[i], y_pred);
    }
    return total / x_test.size();
}

double MLP::calc_accuracy(const std::vector<Tensor<double>>& x_test, const std::vector<Tensor<double>>& y_test) const {
    assert(x_test.size() == y_test.size());
    int correct = 0;
    for (int i = 0; i < (int)x_test.size(); i++) {
        Tensor<double> y_pred = run(x_test[i]);
        if (LinAlg::argmax(y_pred) == LinAlg::argmax(y_test[i])) {
            correct++;
        }
    }
    return (double)correct / x_test.size();
}

void MLP::train(const std::vector<Tensor<double>>& x_train, const std::vector<Tensor<double>>& y_train,
                const std::vector<Tensor<double>>& x_test, const std::vector<Tensor<double>>& y_test, int epochs,
                int batch_size, const std::string& save_file) {
    assert(x_train.size() == y_train.size());
    assert(x_test.size() == y_test.size());
    int num_inputs = x_train.size(), num_layers = layers.size();
    for (int i = 0; i < epochs; i++) {
        std::chrono::time_point<std::chrono::system_clock> start_time = std::chrono::system_clock::now();
        for (int j = 0; j < num_inputs; j += batch_size) {
            progress_bar(j, num_inputs, 69,
                         std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
                                                                               start_time)
                                 .count() /
                             1000.0);
            grad_descent(std::vector<Tensor<double>>(x_train.begin() + j,
                                                     x_train.begin() + std::min(j + batch_size, num_inputs)),
                         std::vector<Tensor<double>>(y_train.begin() + j,
                                                     y_train.begin() + std::min(j + batch_size, num_inputs)));
        }
        progress_bar(num_inputs, num_inputs, 69,
                     std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
                                                                           start_time)
                             .count() /
                         1000.0);
        std::cout << std::endl;
        std::cout << "Epoch " << i << ":\n";
        std::cout << "Train loss: " << calc_loss(x_train, y_train) << "\n";
        std::cout << "Train accuracy: " << calc_accuracy(x_train, y_train) << "\n";
        std::cout << "Test loss: " << calc_loss(x_test, y_test) << "\n";
        std::cout << "Test accuracy: " << calc_accuracy(x_test, y_test) << "\n";
        if (save_file.size() > 0) {
            save(save_file);
        }
    }
}

void MLP::save(std::string filename) const {
    std::ofstream file(filename);
    file << layers.size() << std::endl;
    for (Layers::Layer* l : layers) {
        l->save(file);
    }
}

void MLP::load(std::string filename) {
    std::ifstream file(filename);
    int num_layers;
    file >> num_layers;
    layers.clear();
    for (int i = 0; i < num_layers; i++) {
        layers.push_back(Layers::load(file));
    }
}

void MLP::summary() {
    std::cout << "Layers:\n";
    for (int i = 0; i < (int)layers.size(); i++) {
        std::cout << "Layer " << i << ":\n";
        layers[i]->summary();
    }
}

} // namespace FJML
