#include <cstring>

template<size_t N>
struct sum {
    constexpr static int value = N + sum<N-1>::value;
};

template<>
struct sum<1> {
    constexpr static int value = 1;
};
