cmake_minimum_required (VERSION 2.6)
project (machine-learning)

include_directories ("${PROJECT_SOURCE_DIR}/maths")
add_subdirectory (maths) 

include_directories ("${PROJECT_SOURCE_DIR}/common")
add_subdirectory (common) 

include_directories ("${PROJECT_SOURCE_DIR}/mnist")
add_subdirectory (mnist) 
 
# add the executable
add_executable(NeuralNet neural_network.c mnist/prepare_data.c)
target_link_libraries (NeuralNet maths common)
