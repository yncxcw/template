#include "remove_any.h"

template <class T>
struct is_char {
    constexpr static int value = 0;
};

template <>
struct is_char<char> {
    constexpr static int value = 1;
};

template <class T>
struct is_any_char {
    constexpr static int value = is_char<typename remove_any<T>::type>::value;
};
