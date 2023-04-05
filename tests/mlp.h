#include <catch2/catch_all.hpp>

#include <FJML.h>

using namespace Catch;
using namespace FJML;

TEST_CASE("Test mlp", "[mlp]") {
    MLP mlp({new Layers::Dense(1, 1, Activations::linear)}, Loss::mse);

    SECTION("Test set_loss") {
        mlp.set_loss(Loss::huber);
        REQUIRE(mlp.loss_fn.loss_fn(0, 1) == Approx(0.50));
    }

    // TODO: Complete rest of tests
}
