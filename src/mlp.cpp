// Copyright (c) 2023 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <algorithm>
#include <cstring>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "../include/FJML/mlp.h"

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

void MLP::grad_descent(const Tensor& x_train, const Tensor& y_train) {
    int num_layers = layers.size();

    std::vector<Tensor> run_res(num_layers + 1);
    run_res[0] = x_train;
    for (int i = 0; i < num_layers; i++) {
        run_res[i + 1] = layers[i]->apply(run_res[i]);
    }

    Tensor out_grad = loss_fn.calc_derivative(y_train, run_res[num_layers]);
    for (int i = num_layers - 1; i >= 0; i--) {
        out_grad = layers[i]->backward(run_res[i], out_grad);
    }
}

void MLP::backwards_pass(const Tensor& input, const Tensor& grads) {
    int num_layers = layers.size();

    std::vector<Tensor> run_res(num_layers + 1);
    run_res[0] = input;
    for (int i = 0; i < num_layers; i++) {
        run_res[i + 1] = layers[i]->apply(run_res[i]);
    }

    Tensor out_grad = grads;
    for (int i = num_layers - 1; i >= 0; i--) {
        out_grad = layers[i]->backward(run_res[i], out_grad);
    }
}

Tensor MLP::run(const Tensor& input) const {
    Tensor result = input;
    for (Layers::Layer* l : layers) {
        result = l->apply(result);
    }
    return result;
}

double MLP::calc_loss(const Tensor& x_test, const Tensor& y_test) const {
    return loss_fn.calc_loss(run(x_test), y_test) / x_test.shape[0];
}

double MLP::calc_accuracy(const Tensor& x_test, const Tensor& y_test) const {
    int correct = 0;
    Tensor y_pred = run(x_test);
    for (int i = 0; i < (int)x_test.shape[0]; i++) {
        if (LinAlg::argmax(y_pred, 1, i) == LinAlg::argmax(y_test, 1, i)) {
            correct++;
        }
    }
    return (double)correct / x_test.shape[0];
}

#define time_elapsed                                                                                                   \
    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() /     \
        1000.0

void MLP::train(const Tensor& x_train, const Tensor& y_train, const Tensor& x_test, const Tensor& y_test, int epochs,
                int batch_size, const std::string& save_file) {
    if (x_train.shape[0] != y_train.shape[0]) {
        throw std::invalid_argument("x_train and y_train must have the same number of samples");
    }
    if (x_test.shape[0] != y_test.shape[0]) {
        throw std::invalid_argument("x_test and y_test must have the same number of samples");
    }
    int num_inputs = x_train.shape[0];
    std::vector<int> indices(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        indices[i] = i;
    }
    for (int i = 0; i < epochs; i++) {
        std::cout << "Epoch " << i + 1 << ":\n";
        std::chrono::time_point<std::chrono::system_clock> start_time = std::chrono::system_clock::now();
        std::random_shuffle(indices.begin(), indices.end());
        for (int j = 0; j < num_inputs; j += batch_size) {
            progress_bar(j, num_inputs, 69, time_elapsed);
            int batch_end = std::min(j + batch_size, num_inputs);
            std::vector<int> batch_shape_x = x_train.shape, batch_shape_y = y_train.shape;
            batch_shape_x[0] = batch_end - j;
            batch_shape_y[0] = batch_end - j;
            Tensor x_batch(batch_shape_x, x_train.device);
            Tensor y_batch(batch_shape_y, y_train.device);
            for (int k = j; k < batch_end; k++) {
                memcpy(x_batch.data + (k - j) * x_train.data_size[1], x_train.data + indices[k] * x_train.data_size[1],
                       x_train.data_size[1] * sizeof(double));
                memcpy(y_batch.data + (k - j) * y_train.data_size[1], y_train.data + indices[k] * y_train.data_size[1],
                       y_train.data_size[1] * sizeof(double));
            }
            grad_descent(x_batch, y_batch);
        }
        progress_bar(num_inputs, num_inputs, 69, time_elapsed);
        std::cout << std::endl;
        std::cout << "Training loss: " << calc_loss(x_train, y_train) << std::endl;
        std::cout << "Training accuracy: " << calc_accuracy(x_train, y_train) << std::endl;
        std::cout << "Testing loss: " << calc_loss(x_test, y_test) << std::endl;
        std::cout << "Testing accuracy: " << calc_accuracy(x_test, y_test) << std::endl;
        if (save_file.size() > 0) {
            save(save_file);
        }
    }
}
#undef time_elapsed

void MLP::save(std::string filename) const {
    std::ofstream file(filename);
    file << layers.size() << std::endl;
    for (Layers::Layer* l : layers) {
        l->save(file);
    }
}

void MLP::load(std::string filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "File " << filename << " could not be opened" << std::endl;
        throw std::invalid_argument("File " + filename + " could not be opened");
    }
    std::cout << "Loading model from " << filename << std::endl;
    int num_layers;
    file >> num_layers;
    std::cout << "Number of layers: " << num_layers << std::endl;
    layers.clear();
    for (int i = 0; i < num_layers; i++) {
        layers.push_back(Layers::load(file));
        std::cout << "Loaded layer " << i << ": " << layers[i]->name << std::endl;
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
