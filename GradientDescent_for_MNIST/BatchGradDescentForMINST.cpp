#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <random>


float alpha{0.01};

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<std::vector<uint8_t>> readMNISTImages(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);

        std::vector<std::vector<uint8_t>> images(number_of_images, std::vector<uint8_t>(n_rows * n_cols));
        for (int i = 0; i < number_of_images; i++) {
            file.read((char*)images[i].data(), n_rows * n_cols);
        }
        return images;
    } else {
        throw std::runtime_error("MNIST image file not found!");
    }
}

std::vector<uint8_t> readMNISTLabels(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0, number_of_items = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_items, sizeof(number_of_items));
        number_of_items = reverseInt(number_of_items);

        std::vector<uint8_t> labels(number_of_items);
        file.read((char*)labels.data(), number_of_items);
        return labels;
    } else {
        throw std::runtime_error("MNIST label file not found!");
    }
}


float calculatedY(const std::vector<float>& w, const std::vector<float>& x, float* b) 
{
    if (w.size() != x.size()) {
        throw std::runtime_error("dimensions of w and x must be equal!");
    }
    float y_i = 0.0f;

    for (size_t i = 0; i < w.size(); ++i) {
        y_i += w[i] * x[i];
    }

    if (b != nullptr) {
        y_i += *b;
    }
    return y_i;
}

void UpdateParametersBatch(std::vector<float>& w,
                           const std::vector<std::vector<float>>& X,
                           const std::vector<float>& y,
                           float* b,
                           float alpha)
{
    size_t m = X.size();   
    size_t n = w.size();      

    std::vector<float> grad_w(n, 0.0f);
    float grad_b = 0.0f; 

    for (size_t i = 0; i < m; ++i) {
        float y_hat = calculatedY(w, X[i], b);
        float error = y_hat - y[i];

        for (size_t j = 0; j < n; ++j) {
            grad_w[j] += error * X[i][j]; 
        }

        grad_b += error; 
    }

    for (size_t j = 0; j < n; ++j) {
        w[j] -= alpha * grad_w[j] / m;
    }

    if (b != nullptr) {
        *b -= alpha * grad_b / m;
    }
}

int main() {
    auto images = readMNISTImages("train-images-idx3-ubyte");
    auto labels = readMNISTLabels("train-labels-idx1-ubyte");
    std::vector<std::vector<float>> X(images.size(), std::vector<float>(784));
    for (size_t i = 0; i < images.size(); ++i) {
        for (size_t j = 0; j < 784; ++j) {
            X[i][j] = static_cast<float>(images[i][j]) / 255.0f;
        }
    }
    std::vector<float> Y(labels.size());
    for (size_t i = 0; i < labels.size(); ++i) {
        Y[i] = static_cast<float>(labels[i]);
    }
    std::vector<float> w(784);
    std::mt19937 gen(42); // sabit seed
    std::uniform_real_distribution<float> dis(-0.05f, 0.05f);
    for (auto& wi : w) wi = dis(gen);

    for (size_t i = 0; i < labels.size(); ++i)
        Y[i] = static_cast<float>(labels[i]) / 9.0f;

    float b{1};

    for (int epoch = 0; epoch < 100; ++epoch) {
        UpdateParametersBatch(w, X, Y, &b, alpha);
    }

    std::cout << "After train bias: " << b << std::endl;
    std::cout << "After train w[0]: " << w[0] << std::endl;

    auto test_images = readMNISTImages("t10k-images-idx3-ubyte");
    auto test_labels = readMNISTLabels("t10k-labels-idx1-ubyte");

    std::vector<std::vector<float>> X_test(test_images.size(), std::vector<float>(784));
    for (size_t i = 0; i < test_images.size(); ++i)
        for (size_t j = 0; j < 784; ++j)
            X_test[i][j] = static_cast<float>(test_images[i][j]) / 255.0f;

    std::vector<float> Y_test(test_labels.size());
    for (size_t i = 0; i < test_labels.size(); ++i)
        Y_test[i] = static_cast<float>(test_labels[i]) / 9.0f;

    for (size_t i = 0; i < 10; ++i) { 
        float y_hat = calculatedY(w, X_test[i], &b);
        std::cout << "Test example " << i 
                  << " guess: " << y_hat 
                  << ", real value: " << Y_test[i] << std::endl;
    }

    return 0;
}
