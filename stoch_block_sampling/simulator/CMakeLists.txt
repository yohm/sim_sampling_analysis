cmake_minimum_required(VERSION 3.10)
project(simulator)

set(CMAKE_CXX_STANDARD 11)

include_directories(.)

add_executable(stoch_block_sampling.out
        network.cpp
        network.hpp
        sampling.cpp
        sampling.hpp
        stoch_block_sampling.cpp)
