cmake_minimum_required(VERSION 3.5.1)
project(rNdN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
 
find_package(CUDA 11.3 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/nvvm/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/nvvm/lib64)

find_package(LLVM 14.0 REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_definitions(${LLVM_DEFINITIONS})

find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

include_directories(include)
include_directories(tests)
set(CMAKE_SOURCE_DIR src)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Language)

bison_target(HorseIRParser ${CMAKE_SOURCE_DIR}/HorseIR/Language/horseir.y ${CMAKE_CURRENT_BINARY_DIR}/Language/HorseIR_Parser.cpp)
flex_target(HorseIRScanner ${CMAKE_SOURCE_DIR}/HorseIR/Language/horseir.l ${CMAKE_CURRENT_BINARY_DIR}/Language/HorseIR_Scanner.cpp)
ADD_FLEX_BISON_DEPENDENCY(HorseIRScanner HorseIRParser)

file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/*/*.cpp")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(rNdN_common STATIC ${SOURCES})
add_executable(rNdN ${CMAKE_SOURCE_DIR}/Main.cpp ${BISON_HorseIRParser_OUTPUTS} ${FLEX_HorseIRScanner_OUTPUTS})
add_executable(test_sass ${CMAKE_SOURCE_DIR}/TestSASS.cpp)
add_executable(test_scheduler ${CMAKE_SOURCE_DIR}/TestScheduler.cpp)

target_link_libraries(rNdN rNdN_common cuda ${CUDA_LIBRARIES} libnvptxcompiler_static.a libnvvm.so LLVM LLVMCore LLVMBitWriter LLVMIRReader LLVMPasses libpcre2-8.so)
target_link_libraries(test_sass rNdN_common cuda ${CUDA_LIBRARIES} libnvptxcompiler_static.a libnvvm.so LLVM LLVMCore LLVMBitWriter LLVMIRReader LLVMPasses libpcre2-8.so)
target_link_libraries(test_scheduler rNdN_common cuda ${CUDA_LIBRARIES} libnvptxcompiler_static.a libnvvm.so LLVM LLVMCore LLVMBitWriter LLVMIRReader LLVMPasses libpcre2-8.so)
