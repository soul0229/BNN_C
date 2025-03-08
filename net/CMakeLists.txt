aux_source_directory(${CMAKE_CURRENT_LIST_DIR} NET_SRC)

add_library(net OBJECT ${NET_SRC})

# target_link_libraries(net
#     PRIVATE cjson m  # 私有链接库
# )