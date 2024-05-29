#include <catch2/catch_all.hpp>

#include "../include/FJML/loss.h"

using namespace FJML;
using namespace Catch;

TEST_CASE("Testing loss functions", "[loss]") {
    Tensor y = Tensor::array(std::vector<float>{1, 2, 3}), yhat = Tensor::array(std::vector<float>{3, 2, 1});
    y.reshape({1, 3});
    yhat.reshape({1, 3});
    SECTION("Testing MSE") {
        REQUIRE(Loss::mse.calc_loss(y, yhat) == 8);

        Tensor dy = Loss::mse.calc_derivative(y, yhat);
        REQUIRE(dy.at(0, 0) == 4);
        REQUIRE(dy.at(0, 1) == 0);
        REQUIRE(dy.at(0, 2) == -4);
    }

    SECTION("Testing huber") {
        REQUIRE(Loss::huber.calc_loss(y, yhat) == 6);

        Tensor dy = Loss::huber.calc_derivative(y, yhat);
        REQUIRE(dy.at(0, 0) == 2);
        REQUIRE(dy.at(0, 1) == 0);
        REQUIRE(dy.at(0, 2) == -2);
    }

    SECTION("Testing binary crossentropy") {
        y.at(0, 0) = 0;
        y.at(0, 1) = 0;
        y.at(0, 2) = 1;
        yhat.at(0, 0) = 0.3;
        yhat.at(0, 1) = 0.4;
        yhat.at(0, 2) = 0.3;

        Loss::Loss loss = Loss::binary_crossentropy(false);

        REQUIRE(loss.calc_loss(y, yhat) == Approx(2.0714733720306593));

        Tensor dy = loss.calc_derivative(y, yhat);

        REQUIRE(dy.at(0, 0) == Approx(1 / 0.7));
        REQUIRE(dy.at(0, 1) == Approx(1 / 0.6));
        REQUIRE(dy.at(0, 2) == Approx(-1 / 0.3));
    }

    SECTION("Testing crossentropy") {
        SECTION("Testing from_logits") {
            y.at(0, 0) = 0;
            y.at(0, 1) = 0;
            y.at(0, 2) = 1;
            yhat.at(0, 0) = 1;
            yhat.at(0, 1) = 2;
            yhat.at(0, 2) = 3;

            Loss::Loss loss = Loss::crossentropy(true);

            REQUIRE(loss.calc_loss(y, yhat) == Approx(0.40760594606399536));

            Tensor dy = loss.calc_derivative(y, yhat);

            REQUIRE(dy.at(0, 0) == Approx(0.09003058075904846));
            REQUIRE(dy.at(0, 1) == Approx(0.24472849071025848));
            REQUIRE(dy.at(0, 2) == Approx(-0.334758996963501));

            yhat.at(0, 0) = 1001;
            yhat.at(0, 1) = 1002;
            yhat.at(0, 2) = 1003;

            REQUIRE(loss.calc_loss(y, yhat) == Approx(0.40760594606399536));

            dy = loss.calc_derivative(y, yhat);

            REQUIRE(dy.at(0, 0) == Approx(0.09003058075904846));
            REQUIRE(dy.at(0, 1) == Approx(0.24472849071025848));
            REQUIRE(dy.at(0, 2) == Approx(-0.334758996963501));

            yhat.at(0, 0) = -999;
            yhat.at(0, 1) = -998;
            yhat.at(0, 2) = -997;

            REQUIRE(loss.calc_loss(y, yhat) == Approx(0.40760594606399536));

            dy = loss.calc_derivative(y, yhat);

            REQUIRE(dy.at(0, 0) == Approx(0.09003058075904846));
            REQUIRE(dy.at(0, 1) == Approx(0.24472849071025848));
            REQUIRE(dy.at(0, 2) == Approx(-0.334758996963501));

            yhat.at(0, 0) = 0;
            yhat.at(0, 1) = 0;
            yhat.at(0, 2) = 100;

            REQUIRE(loss.calc_loss(y, yhat) == Approx(0.0).margin(0.0001));

            dy = loss.calc_derivative(y, yhat);

            REQUIRE(dy.at(0, 0) == Approx(0.0).margin(0.0001));
            REQUIRE(dy.at(0, 1) == Approx(0.0).margin(0.0001));
            REQUIRE(dy.at(0, 2) == Approx(0.0).margin(0.0001));

            yhat.at(0, 2) = 0;
            yhat.at(0, 0) = 100;

            REQUIRE(loss.calc_loss(y, yhat) == Approx(100.0).margin(0.001));

            dy = loss.calc_derivative(y, yhat);

            REQUIRE(dy.at(0, 0) == Approx(1).margin(0.001));
            REQUIRE(dy.at(0, 1) == Approx(0).margin(0.001));
            REQUIRE(dy.at(0, 2) == Approx(-1).margin(0.001));
        }

        SECTION("Testing not from_logits") {
            y.at(0, 0) = 0;
            y.at(0, 1) = 0;
            y.at(0, 2) = 1;
            yhat.at(0, 0) = 0.3;
            yhat.at(0, 1) = 0.4;
            yhat.at(0, 2) = 0.3;

            Loss::Loss loss = Loss::crossentropy(false);

            REQUIRE(loss.calc_loss(y, yhat) == Approx(1.2039728164672852));

            Tensor dy = loss.calc_derivative(y, yhat);

            REQUIRE(dy.at(0, 0) == Approx(0));
            REQUIRE(dy.at(0, 1) == Approx(0));
            REQUIRE(dy.at(0, 2) == Approx(-1 / 0.3));
        }
    }

    SECTION("Testing sparse crossentropy") {
        SECTION("Testing from_logits") {
            y = Tensor::array(std::vector<float>{2});
            yhat.at(0, 0) = 1;
            yhat.at(0, 1) = 2;
            yhat.at(0, 2) = 3;

            Loss::Loss loss = Loss::sparse_categorical_crossentropy(true);

            REQUIRE(loss.calc_loss(y, yhat) == Approx(0.40760594606399536));

            Tensor dy = loss.calc_derivative(y, yhat);

            REQUIRE(dy.at(0, 0) == Approx(0.09003058075904846));
            REQUIRE(dy.at(0, 1) == Approx(0.24472849071025848));
            REQUIRE(dy.at(0, 2) == Approx(-0.334758996963501));
        }

        SECTION("Testing not from_logits") {
            y = Tensor::array(std::vector<float>{2});
            yhat.at(0, 0) = 0.3;
            yhat.at(0, 1) = 0.4;
            yhat.at(0, 2) = 0.3;

            Loss::Loss loss = Loss::sparse_categorical_crossentropy(false);

            REQUIRE(loss.calc_loss(y, yhat) == Approx(1.2039728164672852));

            Tensor dy = loss.calc_derivative(y, yhat);

            REQUIRE(dy.at(0, 0) == Approx(0));
            REQUIRE(dy.at(0, 1) == Approx(0));
            REQUIRE(dy.at(0, 2) == Approx(-1 / 0.3));
        }
    }
}
