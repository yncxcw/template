#include <iostream>
#include "include/is_char.h"

int main() {
    std::cout << "int is char " << is_char<int>::value << std::endl;
    std::cout << "char is char " << is_char<char>::value << std::endl;
    std::cout << "const char is char " << is_any_char<const char>::value << std::endl;  
    std::cout << "const int is char " << is_any_char<const int>::value << std::endl; 
    std::cout << "char* is char " << is_any_char<char *>::value << std::endl;
    std::cout << "const char* is char " << is_any_char<const char*>::value << std::endl;
}
