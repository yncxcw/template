#include <iostream>
#include "include/sum.h"

int main() {
    std::cout << "Sum 10 " << sum<10>::value << std::endl;
    std::cout << "Sum 100 " << sum<100>::value << std::endl;
    return 0;
}

