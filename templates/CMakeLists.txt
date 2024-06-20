cmake_minimum_required (VERSION 3.0)

project (Template)

include_directories(include)

add_executable(sum sum.cpp)
add_executable(pow pow.cpp)
add_executable(is_char is_char.cpp)
add_executable(enable_if enable_if.cpp)