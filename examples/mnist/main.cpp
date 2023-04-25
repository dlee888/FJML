#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <FJML.h>

/**
 * @brief Load data from a csv file
 * @param x The input data
 * @param y The output data
 * @param filename The name of the file to load
 */
void load_data(std::vector<FJML::Tensor<double>>& x, std::vector<FJML::Tensor<double>>& y, std::string filename) {
    // Uses data from the kaggle mnist dataset
    // https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
    std::ifstream file(filename);

    // The first line of the file is the header
    std::string line;
    std::getline(file, line);

    // Read the data
    while (std::getline(file, line)) {
        std::stringstream ss(line);

        // The first value is the label
        int label;
        ss >> label;

        // The rest of the values are the pixels
        FJML::Tensor<double> pixels({28 * 28});
        for (int i = 0; i < 28 * 28; i++) {
            int pixel;
            char comma;
            ss >> comma >> pixel;
            pixels.at(i) = pixel / 255.0;
        }

        // Add the data to the vectors
        x.push_back(pixels);
        y.push_back(FJML::Data::one_hot(label, 10));
    }
}

int main() {
    // Load the data
    std::vector<FJML::Tensor<double>> mnist_train_x, mnist_train_y;
    std::vector<FJML::Tensor<double>> mnist_test_x, mnist_test_y;
    load_data(mnist_train_x, mnist_train_y, "mnist_train.csv");
    load_data(mnist_test_x, mnist_test_y, "mnist_test.csv");
    std::cout << "Loaded " << mnist_train_x.size() << " training samples and " << mnist_test_x.size()
              << " testing samples" << std::endl;

    // Split the data into training and validation sets
    std::vector<FJML::Tensor<double>> x_train, y_train;
    std::vector<FJML::Tensor<double>> x_test, y_test;
    FJML::Data::split(mnist_train_x, mnist_train_y, x_train, y_train, x_test, y_test, 0.8);

    // Create the model
    // The model is a simple MLP with 1 hidden layer
    // The input layer has 28 * 28 = 784 neurons
    // The hidden layer has 128 neurons
    // The output layer has 10 neurons
    //
    // The model constructor takes 3 arguments:
    // 1. A vector of layers
    // 2. A loss function
    // 3. An optimizer
    FJML::MLP model({new FJML::Layers::Dense(28 * 28, 128, FJML::Activations::relu),
                     new FJML::Layers::Dense(128, 10, FJML::Activations::linear), new FJML::Layers::Softmax()},
                    FJML::Loss::crossentropy, new FJML::Optimizers::Adam());

    // Train the model
    model.train(x_train, y_train, x_test, y_test, 30, 128, "mnist.fjml");

    // Evaluate the model
    std::cout << "Training accuracy: " << model.calc_accuracy(x_train, y_train) << std::endl;
    std::cout << "Testing accuracy: " << model.calc_accuracy(mnist_test_x, mnist_test_y) << std::endl;
}

// Compile with:
// g++ -std=c++17 -O2 -o main main.cpp -lFJML
