aux_source_directory(${CMAKE_CURRENT_LIST_DIR} CORE_SRC)

add_library(core OBJECT ${CORE_SRC})

# target_link_libraries(core
#     PRIVATE cjson m  # 私有链接库
# )