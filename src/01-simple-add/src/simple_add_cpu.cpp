#include <iostream>
#include <math.h>
#include <chrono>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++) y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<20; // 1M elements

    float *x = new float[N];
    float *y = new float[N];

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    // Run kernel on 1M elements on the CPU
    add(N, x, y);
    auto endTime = std::chrono::high_resolution_clock::now();
    double add_time = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "Addition Duration: " << add_time << "\n";

    // Free memory
    delete [] x;
    delete [] y;

    return 0;
}