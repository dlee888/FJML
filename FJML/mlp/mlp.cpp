// Copyright (c) 2022 David Lee
// This code is licensed under MIT license (see LICENSE for details)

#include <algorithm>

#include "mlp.h"

void progress_bar(int curr, int tot, int bar_width = 69) {
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
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

namespace FJML {

void MLP::grad_descent(std::vector<layer_vals> x_train, std::vector<layer_vals> y_train,
                       std::vector<std::vector<bool>>* mask) {
    assert(x_train.size() == y_train.size());
    int num_inputs = x_train.size(), num_layers = layers.size();

    std::vector<std::vector<layer_vals>> run_res(num_layers + 1);
    run_res[0] = x_train;
    for (int i = 0; i < num_layers; i++) {
        run_res[i + 1] = layers[i]->apply(run_res[i]);
    }

    std::vector<layer_vals> out_grad(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        out_grad[i] = loss_fn.calc_grad(y_train[i], run_res[num_layers][i]);
    }

    for (int i = num_layers - 1; i >= 0; i--) {
        out_grad = layers[i]->apply_grad(run_res[i], out_grad);
    }
}

void MLP::backwards_pass(const std::vector<layer_vals>& input, const std::vector<layer_vals>& grads) {
    int num_layers = layers.size();
    std::vector<std::vector<layer_vals>> run_res(num_layers + 1);
    run_res[0] = input;
    for (int i = 0; i < num_layers; i++) {
        run_res[i + 1] = layers[i]->apply(run_res[i]);
    }

    std::vector<layer_vals> out_grad = grads;

    for (int i = num_layers - 1; i >= 0; i--) {
        out_grad = layers[i]->apply_grad(run_res[i], out_grad);
    }
}

layer_vals MLP::run(layer_vals input) {
    for (Layers::Layer* l : layers) {
        input = l->apply(input);
    }
    return input;
}

double MLP::calc_loss(std::vector<layer_vals> x_test, std::vector<layer_vals> y_test) {
    assert(x_test.size() == y_test.size());
    double total = 0;
    for (int i = 0; i < (int)x_test.size(); i++) {
        layer_vals y_pred = run(x_test[i]);
        total += loss_fn.calc_loss(y_test[i], y_pred);
    }
    return total / x_test.size();
}

double MLP::calc_accuracy(std::vector<layer_vals> x_test, std::vector<layer_vals> y_test) {
    assert(x_test.size() == y_test.size());
    int correct = 0;
    for (int i = 0; i < (int)x_test.size(); i++) {
        layer_vals y_pred = run(x_test[i]);
        if (LinAlg::argmax(y_pred) == LinAlg::argmax(y_test[i])) {
            correct++;
        }
    }
    return (double)correct / x_test.size();
}

void MLP::train(std::vector<layer_vals> x_train, std::vector<layer_vals> y_train, std::vector<layer_vals> x_test,
                std::vector<layer_vals> y_test, int epochs, int batch_size, std::string save_file,
                std::vector<std::vector<bool>>* mask) {
    assert(x_train.size() == y_train.size());
    assert(x_test.size() == y_test.size());
    int num_inputs = x_train.size();
    for (int i = 0; i < epochs; i++) {
        std::vector<int> indices(num_inputs);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_shuffle(indices.begin(), indices.end());
        for (int j = 0; j < num_inputs; j += batch_size) {
            std::vector<layer_vals> x_batch, y_batch;
            std::vector<std::vector<bool>> mask_batch;
            for (int k = j; k < std::min(j + batch_size, num_inputs); k++) {
                x_batch.push_back(x_train[indices[k]]);
                y_batch.push_back(y_train[indices[k]]);
                if (mask != nullptr) {
                    mask_batch.push_back(mask->at(indices[k]));
                }
            }
            progress_bar(j, num_inputs);
            grad_descent(x_batch, y_batch, mask != nullptr ? &mask_batch : nullptr);
        }
        progress_bar(num_inputs, num_inputs);
        std::cout << std::endl;
        std::cout << "Epoch " << i << ":\n";
        std::cout << "Train loss: " << calc_loss(x_train, y_train) << "\n";
        std::cout << "Train accuracy: " << calc_accuracy(x_train, y_train) << "\n";
        std::cout << "Test loss: " << calc_loss(x_test, y_test) << "\n";
        std::cout << "Test accuracy: " << calc_accuracy(x_test, y_test) << "\n";
        if (save_file != "") {
            save(save_file);
        }
    }
}

void MLP::save(std::string filename) {
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
