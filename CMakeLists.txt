cmake_minimum_required(VERSION 3.1)

project(BNN VERSION 0.1 LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # 使能生成 compile_commands.json

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/net)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)