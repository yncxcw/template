#include <type_traits>
#include <iostream>

// check_class only expects to be initialized with class.
template <typename T, typename Enable=void>
struct check_class;

template <typename T>
struct check_class<T, typename std::enable_if<std::is_class<T>::value>::type> {
  static constexpr bool value = true;
};


template<typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
is_even(T v) {
    return v % 2 == 0;
}

template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
bool is_odd(T v) {
    return v % 2 ==1;
}

class A{};

int main() {

    std::cout << "A is class " << check_class<A>::value << std::endl;
    // std::cout << "int is class" << check_class<int>::value << std::endl; // compiler error
    std::cout << "Is even: " << is_even(10) << std::endl;
    std::cout << "Is odd: "  << is_odd(10) << std::endl;
    // TODO: use static_assert for more user-friendly error message.
    // std::cout << "Is even: " << is_even<float>(10.1) << std::endl;  // compiler error 
    // std::cout << "Is odd: "  << is_idd<float>(10.1) << std::endl; // compiler error
}
