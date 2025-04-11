cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(custom_allreduce_op LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
set(CMAKE_CUDA_FLAGS "-lineinfo --threads 4")

find_package(Torch REQUIRED)

# Define our library target
add_library(eecs471 SHARED new_forward.cpp ../new_forward.cu)
target_include_directories(eecs471 PRIVATE .)
# Link against LibTorch
target_link_libraries(eecs471 "${TORCH_LIBRARIES}")

if(LINUX)
	add_library(sandbox SHARED sandbox.cpp)
	target_link_libraries(eecs471 sandbox cudart) # nvidia-ptxjitcompiler
else()
	message(WARN "Sandboxing not available")
endif()
set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
