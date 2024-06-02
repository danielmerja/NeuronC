#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Function to calculate the sigmoid of a value
float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

// Type definition for training data samples
typedef float TrainingSample[3];

// Training datasets for different logical operations
TrainingSample or_train[] = {
    {0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
};

TrainingSample and_train[] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 1}
};

TrainingSample nand_train[] = {
    {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 0}
};

TrainingSample xor_train[] = {
    {0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 0}
};

// Pointer to the currently used training set and its size
TrainingSample *train = and_train;
size_t train_count = 4;

// Function to compute the cost for the current model parameters
float cost(float w1, float w2, float bias) {
    float total_cost = 0.0f;
    for (size_t i = 0; i < train_count; ++i) {
        float prediction = sigmoid(train[i][0] * w1 + train[i][1] * w2 + bias);
        float error = prediction - train[i][2];
        total_cost += error * error;
    }
    return total_cost / train_count;
}

// Function to compute the derivative of the cost function using finite differences
void compute_gradients_finite_difference(float epsilon, float w1, float w2, float bias,
                                         float *dw1, float *dw2, float *db) {
    float current_cost = cost(w1, w2, bias);
    *dw1 = (cost(w1 + epsilon, w2, bias) - current_cost) / epsilon;
    *dw2 = (cost(w1, w2 + epsilon, bias) - current_cost) / epsilon;
    *db = (cost(w1, w2, bias + epsilon) - current_cost) / epsilon;
}

// Function to compute the gradient of the cost function using backpropagation
void compute_gradients_backprop(float w1, float w2, float bias,
                                float *dw1, float *dw2, float *db) {
    *dw1 = 0;
    *dw2 = 0;
    *db = 0;
    for (size_t i = 0; i < train_count; ++i) {
        float input1 = train[i][0];
        float input2 = train[i][1];
        float target = train[i][2];
        float output = sigmoid(input1 * w1 + input2 * w2 + bias);
        float delta = 2 * (output - target) * output * (1 - output);
        *dw1 += delta * input1;
        *dw2 += delta * input2;
        *db += delta;
    }
    *dw1 /= train_count;
    *dw2 /= train_count;
    *db /= train_count;
}

// Function to generate a random float between 0 and 1
float random_float(void) {
    return (float)rand() / (float)RAND_MAX;
}

int main(void) {
    srand(time(NULL));
    float w1 = random_float();
    float w2 = random_float();
    float bias = random_float();

    float learning_rate = 0.1;

    // Training loop
    for (size_t i = 0; i < 10000; ++i) {
        float current_cost = cost(w1, w2, bias);
        printf("Iteration %zu: Cost = %f, w1 = %f, w2 = %f, bias = %f\n", i, current_cost, w1, w2, bias);

        float dw1, dw2, db;
        float epsilon = 0.1;  // Epsilon for finite difference calculation
        compute_gradients_finite_difference(epsilon, w1, w2, bias, &dw1, &dw2, &db);

        w1 -= learning_rate * dw1;
        w2 -= learning_rate * dw2;
        bias -= learning_rate * db;
    }

    printf("Final Cost = %f, w1 = %f, w2 = %f, bias = %f\n", cost(w1, w2, bias), w1, w2, bias);

    // Print the final trained function outputs
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            printf("%zu | %zu = %f\n", i, j, sigmoid(i * w1 + j * w2 + bias));
        }
    }

    return 0;
}
