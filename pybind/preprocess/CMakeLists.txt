
cmake_minimum_required(VERSION 3.4)
project(preprocess)
add_subdirectory(pybind11)


pybind11_add_module(preprocess preprocess.cc)



