# Farmer John.ml

A ML library in C++.

### Why FJML?

- Written completely from scratch
  - Implemented in an easy-to-understand way
  - Completely self-contained, does not rely on libraries
- Both low level and high level
  - Written in C++
  - Includes low level components
  - Includes high level abstractions
- Flexible and easily extensible

### Installation

Simply run `sudo make` to install the library. To build the library
with debug info, use `sudo make debug=true` instead.

To compile programs with FJML, just add the flag `-lFJML` to the end of each compile command.

### Why Farmer John

Elements of this library were inspired by USACO. The library is written with
simple code, in the style that most programmers with little experience outside
of USACO can understand.

### Example

The standard sequential model can be created by creating a new `FJML::MLP`
object, and then adding layers by calling `FJML::MLP::add`. Set the loss and
optimizer by using `FJML::MLP::set_loss` and `FJML::MLP::set_optimizer` (or
specify in the constructor).

The model can then be trained by calling `FJML::MLP::train` and passing in
training and validation data.

```cpp
#include <FJML.h>

int main() {
  FJML::MLP mlp;
  mlp.add(new FJML::Layers::Dense(10, 1, FJML::Activations::linear));
  mlp.set_loss(FJML::Loss::mse);
  mlp.set_optimizer(new FJML::Optimizers::SGD(learning_rate));
  mlp.train(x_train, y_train, x_test, y_test, epochs, batch_size);
}
```

For more examples, see the `examples/` folder.

### Features

Currently has support for:

- Activations:
  - Sigmoid
  - Linear
  - ReLu
  - Swish
  - Leaky ReLu
  - Tanh
- Layers:
  - Dense layers
  - Softmax layers
- Loss Functions:
  - Mean Squared Error
  - Huber
  - Log Loss (Cross Entropy)
- Optimizers:
  - SGD
  - Adam
