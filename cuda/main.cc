#include <iostream>
#include "include/kernel.h"


int main() {
    const size_t N = 10;

    double* ptr_a = new double[N];
    double* ptr_b = new double[N];
    double* ptr_c = new double[N];
   
    for(int i=0; i < N; i++) {
        ptr_a[i] = i * 1.0;
        ptr_b[i] = i * i * 1.0;
    }
    
    // Call to the kernel function
    gpu_add_loop(ptr_a, ptr_b, ptr_c, N);

    for(int i=0; i < N; i++) {
        std::cout << ptr_a[i] << "+" << ptr_b[i] << " = " << ptr_c[i] << std::endl;
    }

    delete ptr_a;
    delete ptr_b;
    delete ptr_c;    
    return 0;
}
