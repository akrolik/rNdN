cmake_minimum_required(VERSION 3.5.1)
project(r3d3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "-O3 -g")
 
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

include_directories(include)
include_directories(tests)
set(CMAKE_SOURCE_DIR src)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/FrontEnd)

bison_target(HorseIRParser ${CMAKE_SOURCE_DIR}/FrontEnd/horseir.y ${CMAKE_CURRENT_BINARY_DIR}/FrontEnd/HorseIR_Parser.cpp)
flex_target(HorseIRScanner ${CMAKE_SOURCE_DIR}/FrontEnd/horseir.l ${CMAKE_CURRENT_BINARY_DIR}/FrontEnd/HorseIR_Scanner.cpp)
ADD_FLEX_BISON_DEPENDENCY(HorseIRScanner HorseIRParser)

file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/*.cpp")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(r3d3 ${SOURCES} ${BISON_HorseIRParser_OUTPUTS} ${FLEX_HorseIRScanner_OUTPUTS})
target_link_libraries(r3d3 cuda ${CUDA_LIBRARIES})
