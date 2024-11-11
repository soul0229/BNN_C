#include <stdint.h>
#include <stdio.h> 
#include <stdlib.h> 
#include <cjson/cJSON.h> 
#include <string.h>
#include "common.h"
#include "conv.h"
#include "core.h"
#include "utils.h"

#define INIT_DIM(dim,num)   do {                                    \
                                for(int i = 0; i < DIM_DEPTH; i++)  \
                                    dim[i] = num;                   \
                            }while(0)
#define ARRAY_SIZE(dim, result) do{                                     \
                                    result = dim[0];                    \
                                    for(int i = 1; i < DIM_DEPTH; i++)  \
                                        result *= dim[i];               \
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
const char *name_order[OMAX] = {[0]="weight", [1]="bias|alpha", [2]="running_mean", [3]="running_var" };
const char *name_typ[] = {
    [TERROR]="error", 
    [NET_ROOT] = "root", 
    [TLAYER] = "layer", 
    [TSHORTCUT] = "shortcut", 
    [TCONV] = "conv", 
    [TBACHNORM] = "bn",
    [TLINER] = "linear",
    };

const char *tabulate[] = {
    "|       |       |       |       |", 
    "|--------------------------------"
    };


static struct net Net = {
    .type = NET_ROOT,
    .child = NULL,
};

/* Recursive calls, must use double Pointers */
static void read_multi_dim_array(cJSON *array, void **data){
    if (array == NULL || data == NULL) {
        dbg_print("read_multi_dim_array arguments is NULL!\n");
        return;
    }

    if(!cJSON_IsArray(array)) {
        *((float *)(*data)) = array->valuedouble;
        dbg_print("%f ", *((float *)(*data)));
        *data += sizeof(float);
        return;
    }
    
    for (int i = 0; i < cJSON_GetArraySize(array); i++) {
        if(cJSON_GetArrayItem(array, i)){
            read_multi_dim_array(cJSON_GetArrayItem(array, i), data);
        }
    }
}

static Tstruct *node_create(Tstruct *dad, Tstruct new, uint32_t dim[DIM_DEPTH], char *name){
    size_t malloc_size = 0;
    uint32_t arry_size;
    ARRAY_SIZE(dim, arry_size);
    dbg_print("name:%s ,arry_size:%d\n", name, arry_size);
    common_t *node = NULL;
    switch(new){
        case TLAYER:
        case TSHORTCUT:
            node = malloc(sizeof(container_t));
            malloc_size = 0;
            break;
        case TCONV:
        case TLINER:
            node = malloc(sizeof(data_info_t));
            malloc_size = (arry_size+dim[0]) * sizeof(float);
            ((data_info_t *)node)->data = malloc(malloc_size);
            ((data_info_t *)node)->len = FLOAT_BYTE;
            memcpy(((data_info_t *)node)->dim, dim, sizeof(uint32_t)*DIM_DEPTH);
            break;
        case TBACHNORM:
            node = malloc(sizeof(data_info_t));
            malloc_size = arry_size * sizeof(float) * 4;
            ((data_info_t *)node)->data = malloc(malloc_size);
            ((data_info_t *)node)->len = FLOAT_BYTE;
            memcpy(((data_info_t *)node)->dim, dim, sizeof(uint32_t)*DIM_DEPTH);
            break;
        default:printf("Tstruct error!\n");return NULL;
    }

    if(!node){
        dbg_print("node create malloc error!\n");
        return NULL;
    }
    dbg_print("malloc ok. size:%ld\n", malloc_size);

    /* new type have parent */
    if(dad){
        /* new type is container */
        if(((common_t *)dad)->type == TLAYER || ((common_t *)dad)->type == TSHORTCUT){
            node->parent = dad;
            node->sibling = ((common_t *)dad)->child;
            ((common_t *)dad)->child = (Tstruct *)node;
            node->type = new;
        }
        /* new type is data */
        else {
            node->parent = dad;
            node->type = new;
        }
    }
    /* new type don't have parent */
    else {
        node->type = new;
        node->parent = (Tstruct*)&Net;
        node->sibling = Net.child;
        Net.child = (Tstruct*)node;
    }

    strcpy(node->name, name);
    return (Tstruct*)node;
}

static common_t *grep_obj_child(common_t *data, char *name){
    common_t *ret = (data?(common_t *)data->child:(common_t *)Net.child);
    while(ret){
        dbg_print("%s %s result:%d\n", ret->name, name, strcmp(ret->name, name) == 0);
        if(strcmp(ret->name, name) == 0)
            return ret;
        ret = (common_t *)ret->sibling;
    }
    return NULL;
}

static data_info_t *get_data_info(char *str, uint32_t dim[DIM_DEPTH]){
    if(str == NULL)
        return NULL;
    int size = 0;
    char string_buff[16],*str_ptr = str;
    common_t *current = NULL, *last_stage = NULL;
    dbg_print("%s\n", str);

    while(*str_ptr && strchr(str_ptr, '.')){
        memset(string_buff, 0x00, sizeof(string_buff));
        size = strchr(str_ptr, '.') - str_ptr;
        strncpy(string_buff, str_ptr, size);
        current = grep_obj_child(last_stage, string_buff); 
        if(!current){
            Tstruct order;
            /* Determine the missing data type. */
            for(order = TERROR; order < TMAX; order++)
                if(strstr(string_buff, name_typ[order]))
                    break;

            /* Process the data in the shortcut. */
            if(order >= TMAX){
                if(strlen(string_buff) == 1){
                    if(last_stage->type == TLAYER)
                        order = TLAYER;
                    else if(last_stage->type == TSHORTCUT)
                        order = atoi(string_buff) + TCONV;
                }
                else {
                    printf("error! %s len:%ld\n", string_buff, strlen(string_buff));
                    return NULL;
                }
            }
            /* Create missing data. */
            current = (common_t *)node_create((Tstruct *)last_stage, order, dim, string_buff);
            if(!current){
                dbg_print("node create error!\n");
                return NULL;
            }
        }
        last_stage = current;
        str_ptr = strchr(str_ptr, '.') + 1;
    }
    return (data_info_t *)current;
}

static void read_json_data_v2(cJSON *json){
    cJSON *sub_arry = json;
    uint32_t dim[DIM_DEPTH];
    INIT_DIM(dim, 1);
    int cnt = 0;
    uint32_t arry_size = 0;
    void *data;
    Tstruct *stage;

    while(sub_arry){
        if(cJSON_IsArray(sub_arry)){
            dim[cnt++] = cJSON_GetArraySize(sub_arry);
        }
        sub_arry = cJSON_GetArrayItem(sub_arry, 0);
    }

    data_info_t *arry_data = get_data_info(json->string, dim);
    if(arry_data == NULL)
        return;

    data_order_t order;
    for(order = OWEIGHT; order < OMAX; order++)
        if(strstr(name_order[order], strrchr(json->string, '.')+1) != NULL)
            break;
    if(order >= OMAX)
        return;

    dbg_print("%d\n", order);
    ARRAY_SIZE(arry_data->dim, arry_size);
    data = arry_data->data + arry_size*order*sizeof(float);
    if(!data){
        dbg_print("arry_data->data ptr is NULL;");
        return;
    }
    dbg_print("dim[%d][%d][%d][%d]\t\t%s\n", arry_data->dim[0], arry_data->dim[1], arry_data->dim[2], arry_data->dim[3], json->string);
    read_multi_dim_array(json, &data);
}

/* Refer to the linux kernel device tree parsing */
static void reverse_nodes(common_t *parent)
{
	common_t *child, *next;

	/* In-depth first */
	child = (common_t *)parent->child;
	while (child) {
		reverse_nodes(child);

		child = (common_t *)child->sibling;
	}

	/* Reverse the nodes in the child list */
	child = (common_t *)parent->child;
	parent->child = NULL;
	while (child) {
		next = (common_t *)child->sibling;

		child->sibling = parent->child;
		parent->child = (Tstruct *)child;
		child = next;
	}
}

static void obj_type_detech_create_v2(cJSON *json) {
    if (json == NULL) {
        return;
    }
    int cnt = 0;
    cJSON *item = json->child;
    while (item) {
        switch (item->type) {
            case cJSON_Array:
                read_json_data_v2(item);
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

static int tab_cnt;
void printf_net_data(common_t *data){
    common_t *stage = (data?data:(common_t *)&Net);

    if((void*)stage == &Net)
        tab_cnt = 0;
    else
        tab_cnt+=8;

    while(stage != NULL){
        switch(stage->type){
            case NET_ROOT:
            case TLAYER:
            case TSHORTCUT:
                printf("%.*s%.*s%.*s%s\n", tab_cnt, tabulate[0], 4, tabulate[1], (int)(8-strlen(stage->name))/2, "    ",  stage->name);
                if(stage->child != NULL)
                    printf_net_data((common_t *)stage->child);
                break;
            case TCONV:
            case TBACHNORM:
            case TLINER:
                printf("%.*s%.*s%.*s%s\n", tab_cnt, tabulate[0], 4, tabulate[1], (int)(8-strlen(stage->name))/2, "    ",  stage->name);
                break;
        default:break;
        }
        stage = (common_t *)stage->sibling;
    }

    if((void*)stage != &Net)
        tab_cnt-=8;
}

void binary_net_data(common_t *data){
    common_t *stage = (data?data:(common_t *)&Net);
    while(stage != NULL){
        switch(stage->type){
            case NET_ROOT:
            case TLAYER:
            case TSHORTCUT:
                if(stage->child != NULL)
                    binary_net_data((common_t *)stage->child);
                break;
            case TCONV:
                data_info_t *d_info = NULL;
                d_info = (data_info_t*)stage;
                float *data = d_info->data;
                if(d_info->len == FLOAT_BYTE && ((common_t *)d_info->parent)->type != NET_ROOT){
                    uint32_t array_size;
                    ARRAY_SIZE(d_info->dim, array_size);
                    uint8_t *binary = malloc((array_size/32 + d_info->dim[0]) * sizeof(float));
                    for(uint16_t dim0 = 0; dim0 < d_info->dim[0]; dim0++){
                        for(uint16_t dim1 = 0; dim1 < d_info->dim[1]; dim1++){
                            for(uint16_t dim2 = 0; dim2 < d_info->dim[2]; dim2++){
                                for(uint16_t dim3 = 0; dim3 < d_info->dim[3]; dim3++){
                                    binary[dim0*(d_info->dim[1]/8)*d_info->dim[2]*d_info->dim[3]+dim2*d_info->dim[3]*(d_info->dim[1]/8)+dim3*(d_info->dim[1]/8)+(dim1/8)] |= \
                                    ((data[dim0*d_info->dim[1]*d_info->dim[2]*d_info->dim[3]+dim1*d_info->dim[2]*d_info->dim[3]+dim2*d_info->dim[3]+dim3]>0)?(0x80>>(dim1%8)):((uint8_t)(0x00)));
                                }
                            }
                        }
                    }
                    memcpy(((void*)binary)+array_size*sizeof(float)/32,((void*)data)+array_size*sizeof(float), d_info->dim[0]*sizeof(float));
                    free(data);
                    d_info->data = binary;
                    d_info->len = BINARY;
                }
                break;
            case TBACHNORM:
            case TLINER:
                break;
        default:break;
        }
        stage = (common_t *)stage->sibling;
    }
}

void printf_appoint_data(char *str){
    if(str == NULL)
        return;
    int size = 0;
    char string_buff[16],*str_ptr = str;
    common_t *current = NULL, *last_stage = NULL;
    dbg_print("%s\n", str);

    while(*str_ptr && strchr(str_ptr, '.')){
        memset(string_buff, 0x00, sizeof(string_buff));
        size = strchr(str_ptr, '.') - str_ptr;
        strncpy(string_buff, str_ptr, size);
        current = grep_obj_child(last_stage, string_buff); 
        if(!current){
            return;
        }
        last_stage = current;
        str_ptr = strchr(str_ptr, '.') + 1;
    }
    if(current){
        printf("%s\n", str);
        uint32_t array_size, offset = 0;
        data_info_t *d_info = (data_info_t *)current;
        ARRAY_SIZE(d_info->dim, array_size);

        data_order_t order;
        for(order = OWEIGHT; order < OMAX; order++)
            if(strstr(name_order[order], strrchr(str, '.')+1) != NULL)
                break;
        if(order >= OMAX)
            return;

        offset = array_size * order;

        switch(d_info->len){
            case UNKNOW:
                wprint("data byte len unknow!\n");
            case BINARY:
                if(offset == 0){
                    printf("binary\n");
                    for(uint32_t i = 0;i < array_size/8; i++){
                        printf("%02x ", ((uint8_t *)d_info->data)[i]);
                        if(i%(d_info->dim[1]/8) == (d_info->dim[1]/8)-1)
                            printf("\n");
                    }
                }
                else{
                    printf("in binary\n");
                    for(uint32_t i = 0;i < array_size; i++){
                        printf("%f ", ((float *)d_info->data)[i+offset/(sizeof(float)*8)]);
                        if(i%16 == 15)
                                printf("\n");
                    }
                }
                break;
            case TWO_BYTE:
                break;
            case FLOAT_BYTE:
                dbg_print("FLOAT_BYTE\n");
                if (d_info->data != NULL) {
                    float *data = (float *)d_info->data;
                    printf("d_info->data array_size:%d\n", array_size);
                    for(uint32_t i = 0;i < 64*9; i++){
                        printf("%-6f ", data[i+offset]);
                        if(i%9 == 8)
                            printf("\n");
                    }
                }
                else
                    eprint("d_info->data = NULL\n");
                break;
            default:
                eprint("data byte len error!\n");
                break;
        }
    }
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
    reverse_nodes((common_t *)&Net);
    printf_net_data(NULL);
    binary_net_data(NULL);
    printf_appoint_data("layer1.0.conv1.weight");

    // 释放 cJSON 结构体并释放内存
    cJSON_Delete(root);
    free(json_buffer);
// 55 fe 12 43 04 55 5b 15 
    return;
}