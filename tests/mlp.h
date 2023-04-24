#include <catch2/catch_all.hpp>

#include "../src/FJML/mlp/mlp.h"

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
        Tensor<double> input = Tensor<double>::array(std::vector<double>{6.9});
        Tensor<double> output = mlp.run(input);
        REQUIRE(output.shape == std::vector<int>{1});
        REQUIRE(output.at(0) == Approx(12.8));
    }

    SECTION("Test linear regression") {
        mlp.set_loss(Loss::huber);
        double learning_rate = 0.005;
        std::vector<Tensor<double>> x_train, y_train;
        for (int i = -32; i < 33; i++) {
            x_train.push_back(Tensor<double>::array(std::vector<double>{(double)i}));
            y_train.push_back(Tensor<double>::array(std::vector<double>{2.0 * i - 1}));
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
}
