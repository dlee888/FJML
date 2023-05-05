#include <catch2/catch_all.hpp>

#include "../include/FJML/mlp.h"

using namespace Catch;
using namespace FJML;

TEST_CASE("Test mlp", "[mlp]") {
    MLP mlp({new Layers::Dense(1, 1, Activations::linear)}, Loss::mse, new Optimizers::SGD(0.003));

    SECTION("Test set_loss") {
        mlp.set_loss(Loss::huber);
        REQUIRE(mlp.loss_fn.function(0, 1) == Approx(0.50));
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
        Tensor output = mlp.run(input);
        REQUIRE(output.shape == std::vector<int>{1});
        REQUIRE(output.at(0) == Approx(12.8));
    }

    SECTION("Test calc_loss") {
        ((Layers::Dense*)mlp.layers.at(0))->weights.at(0, 0) = 2;
        ((Layers::Dense*)mlp.layers.at(0))->bias.at(0) = -1;
        mlp.set_loss(Loss::huber);

        Tensor input = Tensor::array(std::vector<double>{6.9});
        Tensor output = Tensor::array(std::vector<double>{12.8});
        REQUIRE(mlp.calc_loss({input}, {output}) == Approx(0));

        output = Tensor::array(std::vector<double>{12.9});
        REQUIRE(mlp.calc_loss({input}, {output}) == Approx(0.005));

        output = Tensor::array(std::vector<double>{12.7});
        REQUIRE(mlp.calc_loss({input}, {output}) == Approx(0.005));

        output = Tensor::array(std::vector<double>{11.8});
        REQUIRE(mlp.calc_loss({input}, {output}) == Approx(0.5));
    }

    SECTION("Test calc_accuracy") {
        MLP mlp2({new Layers::Softmax()}, Loss::mse, new Optimizers::SGD(0.003));

        std::vector<Tensor> input, output;
        input.push_back(Tensor::array(std::vector<double>{1, 2, 3}));
        input.push_back(Tensor::array(std::vector<double>{4, 5, 6}));
        input.push_back(Tensor::array(std::vector<double>{7, 8, -9}));
        output.push_back(Tensor::array(std::vector<double>{0, 1, 0}));
        output.push_back(Tensor::array(std::vector<double>{0, 0, 1}));
        output.push_back(Tensor::array(std::vector<double>{1, 0, 0}));

        REQUIRE(mlp2.calc_accuracy(input, output) == Approx(1.0 / 3));
    }

    SECTION("Test linear regression") {
        mlp.set_loss(Loss::huber);
        double learning_rate = 0.005;
        std::vector<Tensor> x_train, y_train;
        for (int i = -32; i < 33; i++) {
            x_train.push_back(Tensor::array(std::vector<double>{(double)i}));
            y_train.push_back(Tensor::array(std::vector<double>{2.0 * i - 1}));
        }
        for (int i = 0; i < 5000; i++) {
            if (i % 10 == 9) {
                learning_rate *= 0.995;
                mlp.set_optimizer(new Optimizers::SGD(learning_rate));
            }
            mlp.grad_descent(x_train, y_train);
        }
        REQUIRE(((Layers::Dense*)mlp.layers.at(0))->weights.at(0, 0) == Approx(2.0).margin(0.001));
        REQUIRE(((Layers::Dense*)mlp.layers.at(0))->bias.at(0) == Approx(-1).margin(0.001));
        REQUIRE(mlp.calc_loss(x_train, y_train) == Approx(0).margin(0.001));
    }

    SECTION("Test logistic regression") {
        MLP mlp2({new Layers::Dense(1, 2), new Layers::Softmax()}, Loss::crossentropy, new Optimizers::Adam());

        std::vector<Tensor> x_train, y_train;
        for (int i = -32; i < 33; i++) {
            x_train.push_back(Tensor::array(std::vector<double>{(double)i}));
            y_train.push_back(Tensor::array(std::vector<double>{i < 0 ? 1.0 : 0.0, i >= 0 ? 1.0 : 0.0}));
        }
        for (int i = 0; i < 5000; i++) {
            mlp2.grad_descent(x_train, y_train);
        }
        REQUIRE(mlp2.calc_accuracy(x_train, y_train) == Approx(1.0).margin(0.001));
    }
}
