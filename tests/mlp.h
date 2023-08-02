#include <catch2/catch_all.hpp>

#include "../include/FJML/mlp.h"

using namespace Catch;
using namespace FJML;

TEST_CASE("Test mlp", "[mlp]") {
    MLP::MLP mlp({new Layers::Dense(1, 1, Activations::linear)}, Loss::mse, new Optimizers::SGD(0.003));

    SECTION("Test set_loss") {
        mlp.set_loss(Loss::huber);
        REQUIRE(mlp.loss_fn.name == "huber");
    }

    SECTION("Test set_optimizer") {
        mlp.set_optimizer(new Optimizers::Adam());
        REQUIRE(((Layers::Dense*)mlp.layers.at(0))->w_opt->name == "Adam");
        REQUIRE(((Layers::Dense*)mlp.layers.at(0))->b_opt->name == "Adam");
    }

    SECTION("Test add") {
        mlp.add(new Layers::Softmax());
        REQUIRE(mlp.layers.size() == 2);
        REQUIRE(mlp.layers.at(1)->name == "Softmax");
    }

    SECTION("Test run") {
        ((Layers::Dense*)mlp.layers.at(0))->weights.at(0, 0) = 2;
        ((Layers::Dense*)mlp.layers.at(0))->bias.at(0) = -1;
        Tensor input = Tensor::array(std::vector<double>{6.9});
        input.reshape(std::vector<int>{1, 1});
        Tensor output = mlp.run(input);
        REQUIRE(output.shape == std::vector<int>{1, 1});
        REQUIRE(output.at(0, 0) == Approx(12.8));
    }

    SECTION("Test grad descent") {
        mlp.set_optimizer(new Optimizers::SGD(0.01));

        ((Layers::Dense*)mlp.layers.at(0))->weights.at(0, 0) = 2;
        ((Layers::Dense*)mlp.layers.at(0))->bias.at(0) = -1;

        Tensor input = Tensor::array(std::vector<double>{6.9});
        input.reshape(std::vector<int>{1, 1});
        Tensor output = Tensor::array(std::vector<double>{12.8});
        output.reshape(std::vector<int>{1, 1});

        mlp.grad_descent(input, output);
        REQUIRE(((Layers::Dense*)mlp.layers.at(0))->weights.at(0, 0) == Approx(2).margin(0.00001));
        REQUIRE(((Layers::Dense*)mlp.layers.at(0))->bias.at(0) == Approx(-1).margin(0.00001));

        output.at(0, 0) = 12.9;
        mlp.grad_descent(input, output);
        REQUIRE(((Layers::Dense*)mlp.layers.at(0))->weights.at(0, 0) == Approx(2.0138).margin(0.000001));
        REQUIRE(((Layers::Dense*)mlp.layers.at(0))->bias.at(0) == Approx(-0.998).margin(0.000001));
    }

    SECTION("Test linear regression") {
        mlp.set_loss(Loss::huber);
        double learning_rate = 0.005;
        std::vector<Tensor> x_train_v, y_train_v;
        for (int i = -32; i < 33; i++) {
            x_train_v.push_back(Tensor::array(std::vector<double>{(double)i}));
            y_train_v.push_back(Tensor::array(std::vector<double>{2.0 * i - 1}));
        }
        Tensor x_train = Tensor::array(x_train_v);
        Tensor y_train = Tensor::array(y_train_v);
        for (int i = 0; i < 5000; i++) {
            if (i % 10 == 9) {
                learning_rate *= 0.995;
                mlp.set_optimizer(new Optimizers::SGD(learning_rate));
            }
            mlp.grad_descent(x_train, y_train);
        }
        REQUIRE(((Layers::Dense*)mlp.layers.at(0))->weights.at(0, 0) == Approx(2.0).margin(0.001));
        REQUIRE(((Layers::Dense*)mlp.layers.at(0))->bias.at(0) == Approx(-1).margin(0.001));
    }

    SECTION("Test logistic regression") {
        MLP::MLP mlp2({new Layers::Dense(1, 2), new Layers::Softmax()}, Loss::crossentropy, new Optimizers::Adam());

        std::vector<Tensor> x_train_v, y_train_v;
        for (int i = -32; i < 33; i++) {
            x_train_v.push_back(Tensor::array(std::vector<double>{(double)i}));
            y_train_v.push_back(Tensor::array(std::vector<double>{i < 0 ? 1.0 : 0.0, i >= 0 ? 1.0 : 0.0}));
        }
        Tensor x_train = Tensor::array(x_train_v);
        Tensor y_train = Tensor::array(y_train_v);
        for (int i = 0; i < 5000; i++) {
            mlp2.grad_descent(x_train, y_train);
        }
        REQUIRE(MLP::accuracy.compute(y_train, mlp2.run(x_train)) == Approx(1.0).margin(0.001));
    }
}
