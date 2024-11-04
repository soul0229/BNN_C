aux_source_directory(${CMAKE_CURRENT_LIST_DIR} BNN_SRC)

add_library(BnnSrc ${BNN_SRC})

target_link_libraries(BnnSrc
    PRIVATE cjson m  # 私有链接库
)