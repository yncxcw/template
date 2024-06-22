#include "include/pow.h"

#include <iostream>

int main() {
    std::cout << pow<5>()(2.0) << std::endl;
    std::cout << pow<1>()(2.0) << std::endl;
    std::cout << pow<4>()(2.0) << std::endl;
}
