set(TEST_SRC "./BConvTest.c")
if (CONFIG_TEST_ENABLE)
    aux_source_directory(${CMAKE_CURRENT_LIST_DIR} TEST_SRC)
    add_library(test OBJECT ${TEST_SRC})
endif()



# target_link_libraries(net
#     PRIVATE cjson m  # 私有链接库
# )