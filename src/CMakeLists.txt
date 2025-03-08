add_subdirectory(core)
add_subdirectory(func)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(${PROJECT_NAME} main.c)
target_link_libraries(${PROJECT_NAME} core func net)
if (CONFIG_TEST_ENABLE)
    target_link_libraries(${PROJECT_NAME} test)
endif()