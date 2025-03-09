aux_source_directory(${CMAKE_CURRENT_LIST_DIR} FUNC_SRC)

add_library(func OBJECT ${FUNC_SRC})

target_link_libraries(func
    PRIVATE cjson m  jpeg# 私有链接库
)