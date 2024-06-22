#include <cstring>

template <size_t N, bool odd = N % 2>
struct pow;

template <size_t N>
struct pow<N, true> {
    double operator()(double x) const { return x * pow<(N - 1) / 2>()(x) * pow<(N - 1) / 2>()(x); }
};

template <size_t N>
struct pow<N, false> {
    double operator()(double x) const { return pow<N / 2>()(x) * pow<N / 2>()(x); }
};

template <>
struct pow<1, true> {
    double operator()(double x) const { return x; }
};

template <>
struct pow<0, false> {
    double operator()(double x) const { return 1; }
};
