# set( CMAKE_VERBOSE_MAKEFILE on )

cmake_minimum_required (VERSION 2.6)
project (machine-learning)
#set(CMAKE_BUILD_TYPE Debug)

include_directories ("${PROJECT_SOURCE_DIR}/maths")
add_subdirectory (maths) 

include_directories ("${PROJECT_SOURCE_DIR}/common")
add_subdirectory (common) 

include_directories ("${PROJECT_SOURCE_DIR}/mnist")
add_subdirectory (mnist) 
 
# add the executable
add_executable(NeuralNet neural_network.c mnist/prepare_data.c)
target_link_libraries (NeuralNet maths common)

add_definitions ("-Wall")

message("CMAKE_COMPILER_IS_GNUCC (${CMAKE_COMPILER_IS_GNUCC}) CMAKE_COMPILER_IS_GNUCXX (${CMAKE_COMPILER_IS_GNUCXX})")
message("CMAKE_C_FLAGS (${CMAKE_C_FLAGS}) CMAKE_C_COMPILER_ID (${CMAKE_C_COMPILER_ID}) CMAKE_CXX_COMPILER_ID(${CMAKE_CXX_COMPILER_ID})")
