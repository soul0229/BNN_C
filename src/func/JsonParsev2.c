#include <stdint.h>
#include <stdio.h> 
#include <stdlib.h> 
#include <cjson/cJSON.h> 
#include <string.h>
#include "core.h"
#include "utils.h"

#define INIT_DIM(dim,num)   do {                                    \
                                for(int i = 0; i < DIM_DEPTH; i++)  \
                                    dim[i] = num;                   \
                            }while(0)

static struct net Net;
static uint32_t dim[DIM_DEPTH];

static void read_multi_dim_array(cJSON *array, void **data){
    if (array == NULL) {
        return;
    }
    if(!cJSON_IsArray(array)) {
        *(float *)(*data) = array->valuedouble;
        dbg_print("%f ", *(float *)(*data));
        *data += sizeof(float);
        return;
    }
    
    for (int i = 0; i < cJSON_GetArraySize(array); i++) {
        if(cJSON_GetArrayItem(array, i)){
            read_multi_dim_array(cJSON_GetArrayItem(array, i), data);
        }
    }
}

static void read_json_data_v2(cJSON *json, object_t **obj_t){
    cJSON *sub_arry = json;
    INIT_DIM(dim, 1);
    int cnt = 0;
    void *data;

    while(sub_arry){
        if(cJSON_IsArray(sub_arry)){
            dim[cnt++] = cJSON_GetArraySize(sub_arry);
        }
        sub_arry = cJSON_GetArrayItem(sub_arry, 0);
    }

    uint32_t arry_size = dim[0]*dim[1]*dim[2]*dim[3];
    data_info_t *arry_data = malloc(sizeof(data_info_t)+arry_size*sizeof(float));
    memcpy(arry_data->dim, dim, sizeof(dim));
    strcpy(arry_data->name, strrchr(json->string, '.')+1);

    arry_data->data = arry_data + sizeof(data_info_t);
    data = arry_data->data;
    read_multi_dim_array(json, &data);
    int (*arry)[dim[0]];
    for(int i=0;i < arry_size;i++)
        printf("%f ", ((float*)arry_data->data)[i]);

    printf("dim[%d][%d][%d][%d]\t\t%s\n", dim[0], dim[1], dim[2], dim[3], json->string);
}

void obj_type_detech_create_v2(cJSON *json, object_t **obj) {
    if (json == NULL) {
        return;
    }
    cJSON *item = json->child;
    while (item) {
        switch (item->type) {
            case cJSON_Array:
                printf("%s\t\t\t\tarry\n", item->string);
                read_json_data_v2(item, obj);
                return;
                break;
            case cJSON_Object:
                obj_type_detech_create_v2(item, obj);
                break;
            default:
                break;
        }
        item = item->next;
    }
    return;
}

void json_model_parse_v2(char* file_name) {
    // 打开并读取 model.json 文件
    FILE *fp = fopen(file_name, "r");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    // 获取文件大小并读取文件内容
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *json_buffer = (char *)malloc(file_size + 1);
    fread(json_buffer, 1, file_size, fp);
    json_buffer[file_size] = '\0';  // 确保以 null 结尾

    fclose(fp);

    // 使用 cJSON 解析 JSON 数据
    cJSON *root = cJSON_Parse(json_buffer);
    if (root == NULL) {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            fprintf(stderr, "Error before: %s\n", error_ptr);
        }
        return;
    }
    memset(&Net, 0x00, sizeof(struct net));
    obj_type_detech_create_v2(root, &Net.start);

    // 释放 cJSON 结构体并释放内存
    cJSON_Delete(root);
    free(json_buffer);

    return;
}