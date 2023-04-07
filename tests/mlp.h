#include <catch2/catch_all.hpp>

#include <FJML.h>

using namespace Catch;
using namespace FJML;

TEST_CASE("Test mlp", "[mlp]") {
    MLP mlp({new Layers::Dense(1, 1, Activations::linear)}, Loss::mse, new Optimizers::SGD<1>(0.003));

    SECTION("Test set_loss") {
        mlp.set_loss(Loss::huber);
        REQUIRE(mlp.loss_fn.loss_fn(0, 1) == Approx(0.50));
    }

    SECTION("Test set_optimizer") {
        mlp.set_optimizer(new Optimizers::Adam<1>());
        REQUIRE(((Layers::Dense*)mlp.layers[0])->w_opt->name == "Adam");
        REQUIRE(((Layers::Dense*)mlp.layers[0])->b_opt->name == "Adam");
    }

    SECTION("Test add") {
        mlp.add(new Layers::Softmax());
        REQUIRE(mlp.layers.size() == 2);
        REQUIRE(mlp.layers[1]->name == "Softmax");
    }

    SECTION("Test run") {
        ((Layers::Dense*)mlp.layers[0])->layer_weights[0][0] = 2;
        ((Layers::Dense*)mlp.layers[0])->layer_bias[0] = -1;
        layer_vals input = layer_vals{1};
        input[0] = 6.9;
        layer_vals output = mlp.run(input);
        REQUIRE(output.size() == 1);
        REQUIRE(output[0] == Approx(12.8));
    }

    SECTION("Test linear regression") {
        mlp.set_loss(Loss::huber);
        double learning_rate = 0.005;
        std::vector<layer_vals> x_train, y_train;
        for (int i = -32; i < 33; i++) {
            layer_vals temp{1};
            temp[0] = i;
            x_train.push_back(temp);
            temp[0] = 2 * i - 1;
            y_train.push_back(temp);
        }
        for (int i = 0; i < 5000; i++) {
            if (i % 10 == 9) {
                learning_rate *= 0.995;
                mlp.set_optimizer(new Optimizers::SGD<1>(learning_rate));
            }
            mlp.grad_descent(x_train, y_train);
        }
        REQUIRE(((Layers::Dense*)mlp.layers[0])->layer_weights[0][0] == Approx(2.0).margin(0.001));
        REQUIRE(((Layers::Dense*)mlp.layers[0])->layer_bias[0] == Approx(-1).margin(0.001));
        REQUIRE(mlp.calc_loss(x_train, y_train) == Approx(0).margin(0.001));
    }
}

