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


#define WEIGHT "weight"
#define BIAS "bias"
#define MEAN "running_mean"
#define VAR "running_var"

typedef enum data_order{
    OWEIGHT = 0,
    OBIAS,
    OMEAN,
    OVAR,
    OMAX,
}data_order_t;
const char *name_order[OMAX] = {"weight", "bias", "running_mean", "running_var" };
const char *name_typ[] = {
    [TERROR]="error", 
    [NET_ROOT] = "root", 
    [TLAYER] = "layer", 
    [TSHORTCUT] = "shortcut", 
    [TCONV] = "conv", 
    [TBACHNORM] = "bn",
    [TLINER] = "linear",
    };

static struct net Net = {
    .type = NET_ROOT,
    .child = NULL,
};
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


Tstruct *node_create(Tstruct *dad, Tstruct new, int arry_size, char *name){
    int malloc_size;
    common_t *node = NULL;
    switch(new){
        case TLAYER:
        case TSHORTCUT:
            malloc_size = sizeof(container_t);
            break;
        case TCONV:
            malloc_size = arry_size * sizeof(float) + sizeof(data_info_t);
            break;
        case TBACHNORM:
            malloc_size = arry_size * sizeof(float)*4 + sizeof(data_info_t);
            break;
        case TLINER:
            malloc_size = arry_size * sizeof(float)*2 + sizeof(data_info_t);
            break;
        default:return NULL;
    }

    /* new type have parent */
    if(dad){
        /* new type is container */
        if(((common_t *)dad)->type == TLAYER || ((common_t *)dad)->type == TSHORTCUT){
            node = malloc(malloc_size);
            node->parent = dad;
            node->sibling = ((common_t *)dad)->child;
            ((common_t *)dad)->child = (Tstruct *)node;
            node->type = new;
        }
        /* new type is data */
        else {  
            node = malloc(malloc_size);
            node->parent = dad;
            node->type = new;
            ((data_info_t*)node)->data = ((void*)node) + sizeof(data_info_t);
        }
    }
    /* new type don't have parent */
    else {
        node = malloc(malloc_size);
        node->type = new;
        node->parent = (Tstruct*)&Net;
        node->sibling = NULL;
        Net.child = (Tstruct*)node;
    }
    strcpy(node->name, name);
    return (Tstruct*)node;
}

common_t *grep_obj_depth(common_t *data, char *name, int aim_depth, int current_depth){
    common_t *ret, *sibling = (data?data:(common_t *)Net.child);
    while(sibling){
        if(aim_depth == current_depth && !strcmp(sibling->name, name))
            return sibling;
        else {
            if(current_depth < aim_depth){
                ret = grep_obj_depth(sibling, name, aim_depth, current_depth+1);
                if(ret)
                    return  ret;
            }
        }
        sibling = (common_t *)sibling->sibling;
    }
    return NULL;
}
#define grep_obj(name, aim_depth) grep_obj_depth(NULL, name, aim_depth, 0)

data_info_t *get_data_info(char *str, uint32_t arry_size){
    if(str == NULL)
        return NULL;
    int depth = 0, size = 0;
    char string_buff[16],*str_ptr = str;
    common_t *current, *last_stage;
    
    while(strchr(str_ptr, '.')){
        memset(string_buff, 0x00, sizeof(string_buff));
        size = strchr(str_ptr, '.') - str_ptr;
        strncpy(string_buff, str_ptr, size);
        current = grep_obj(string_buff, depth); 
        if(!current){
            Tstruct order;
            for(order = TERROR; order < TMAX; order++)
                if(strstr(string_buff, name_typ[order]))
                    break;
            current = (common_t *)node_create((Tstruct *)last_stage, order, arry_size, string_buff);
        }
        last_stage = current;
        str_ptr += (size+1);
        depth++;
    }

    return (data_info_t *)current;
}

static void read_json_data_v2(cJSON *json){
    cJSON *sub_arry = json;
    INIT_DIM(dim, 1);
    int cnt = 0;
    void *data;
    Tstruct *stage;

    while(sub_arry){
        if(cJSON_IsArray(sub_arry)){
            dim[cnt++] = cJSON_GetArraySize(sub_arry);
        }
        sub_arry = cJSON_GetArrayItem(sub_arry, 0);
    }

    uint32_t arry_size = dim[0]*dim[1]*dim[2]*dim[3];
    data_info_t *arry_data = get_data_info(json->string, arry_size);
    memcpy(arry_data->dim, dim, sizeof(dim));
    
    data_order_t order;
    for(order = OWEIGHT; order <= OMAX; order++)
        if(strcmp(strrchr(json->string, '.')+1, name_order[order]))
            break;

    data = arry_data->data + arry_size*order;
    read_multi_dim_array(json, &data);

    dbg_print("dim[%d][%d][%d][%d]\t\t%s\n", dim[0], dim[1], dim[2], dim[3], json->string);
}

void obj_type_detech_create_v2(cJSON *json) {
    if (json == NULL) {
        return;
    }
    cJSON *item = json->child;
    while (item) {
        switch (item->type) {
            case cJSON_Array:
                printf("%s\t\t\t\tarry\n", item->string);
                read_json_data_v2(item);
                return;
                break;
            case cJSON_Object:
                obj_type_detech_create_v2(item);
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

    obj_type_detech_create_v2(root);

    // 释放 cJSON 结构体并释放内存
    cJSON_Delete(root);
    free(json_buffer);

    return;
}