template<class T>
struct remove_any {
    using type = T;
};

template<class T>
struct remove_any<const T> {
    using type = T;
};

template<class T>
struct remove_any<T*> {
    using type = T;
};

template<class T>
struct remove_any<const T*> {
    using type = T;
};

template<class T>
struct remove_any<T&> {
    using type = T;
};

template<class T>
struct remove_any<const T&> {
    using type = T;
};
