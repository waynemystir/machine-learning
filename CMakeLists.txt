cmake_minimum_required (VERSION 2.6)
project (deep-learning)

include_directories ("${PROJECT_SOURCE_DIR}/maths")
add_subdirectory (maths) 
 
# add the executable
add_executable(NeuralNet neural_network.c)
target_link_libraries (NeuralNet maths)
