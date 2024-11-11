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
    if (array == NULL || data == NULL) {
        dbg_print("read_multi_dim_array arguments is NULL!\n");
        return;
    }

    if(!cJSON_IsArray(array)) {
        (*((float *)(*data))) = array->valuedouble;
        // dbg_print("%f ", array->valuedouble);
        *data += sizeof(float);
        return;
    }
    
    for (int i = 0; i < cJSON_GetArraySize(array); i++) {
        if(cJSON_GetArrayItem(array, i)){
            read_multi_dim_array(cJSON_GetArrayItem(array, i), data);
        }
    }
}

const char *tabulate[] = {
    "|                                ", 
    "|--------------------------------"
    };
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
                printf("%.*s%.*s%s\n", tab_cnt, tabulate[0], 4, tabulate[1], stage->name);
                if(stage->child != NULL)
                    printf_net_data((common_t *)stage->child);
                break;
            case TCONV:
            case TBACHNORM:
            case TLINER:
                printf("%.*s%.*s%s\n", tab_cnt, tabulate[0], 4, tabulate[1], stage->name);
                break;
        default:break;
        }
        stage = (common_t *)stage->sibling;
    }

    if((void*)stage != &Net)
        tab_cnt-=8;
}

Tstruct *node_create(Tstruct *dad, Tstruct new, int arry_size, char *name){
    size_t malloc_size = 0;
    common_t *node = NULL;
    switch(new){
        case TLAYER:
        case TSHORTCUT:
            malloc_size = sizeof(container_t);
            break;
        case TCONV:
            malloc_size = (arry_size+512) * sizeof(float) + sizeof(data_info_t);
            break;
        case TBACHNORM:
            malloc_size = arry_size * sizeof(float)*4 + sizeof(data_info_t);
            break;
        case TLINER:
            malloc_size = arry_size * sizeof(float)*2 + sizeof(data_info_t);
            break;
        default:printf("Tstruct error!\n");return NULL;
    }
    /* new type have parent */
    if(dad){
        /* new type is container */
        if(((common_t *)dad)->type == TLAYER || ((common_t *)dad)->type == TSHORTCUT){
            dbg_print("container malloc.\n");
            node = malloc(malloc_size);
            if(!node){
                dbg_print("node create malloc error!\n");
                return NULL;
            }
            dbg_print("container malloc ok. size:%ld\n", malloc_size);
            node->parent = dad;
            node->sibling = ((common_t *)dad)->child;
            ((common_t *)dad)->child = (Tstruct *)node;
            node->type = new;
        }
        /* new type is data */
        else {
            dbg_print("data malloc.\n");
            node = malloc(malloc_size);
            if(!node){
                dbg_print("node create malloc error!\n");
                return NULL;
            }
            dbg_print("data malloc ok. size:%ld\n", malloc_size);
            node->parent = dad;
            node->type = new;
        }
    }
    /* new type don't have parent */
    else {
        dbg_print("root child malloc. size:%ld\n", malloc_size);
        node = malloc(malloc_size);
        if(!node){
            dbg_print("node create malloc error!\n");
            return NULL;
        }
        dbg_print("root child malloc ok. size:%ld\n", malloc_size);
        node->type = new;
        node->parent = (Tstruct*)&Net;
        node->sibling = Net.child;
        Net.child = (Tstruct*)node;
    }

    strcpy(node->name, name);
    return (Tstruct*)node;
}

common_t *grep_obj_child(common_t *data, char *name){
    common_t *ret = (data?(common_t *)data->child:(common_t *)Net.child);
    while(ret){
        dbg_print("%s %s result:%d\n", ret->name, name, strcmp(ret->name, name) == 0);
        if(strcmp(ret->name, name) == 0)
            return ret;
        ret = (common_t *)ret->sibling;
    }
    return NULL;
}

data_info_t *get_data_info(char *str, uint32_t arry_size){
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
            current = (common_t *)node_create((Tstruct *)last_stage, order, arry_size, string_buff);
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
    if(arry_data == NULL)
        return;
    memcpy(arry_data->dim, dim, sizeof(dim));

    data_order_t order;
    for(order = OWEIGHT; order < OMAX; order++)
        if(strcmp(strrchr(json->string, '.')+1, name_order[order]))
            break;

    data = arry_data->data + arry_size*order*sizeof(float);
    data = arry_data->data;
    if(!data){
        dbg_print("arry_data->data ptr is NULL;");
        return;
    }
    dbg_print("dim[%d][%d][%d][%d]\t\t%s\n", dim[0], dim[1], dim[2], dim[3], json->string);
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

void obj_type_detech_create_v2(cJSON *json) {
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

    // 释放 cJSON 结构体并释放内存
    cJSON_Delete(root);
    free(json_buffer);

    return;
}