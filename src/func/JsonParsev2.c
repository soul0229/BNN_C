#include <stdint.h>
#include <stdio.h> 
#include <stdlib.h> 
#include <cjson/cJSON.h> 
#include <string.h>
#include <threads.h>
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
net_t *net_start;

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
        node->parent = dad;
        node->sibling = ((common_t *)dad)->child;
        ((common_t *)dad)->child = (Tstruct *)node;
        node->type = new;
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
void printf_net_structure(common_t *data){
    common_t *stage = (data?data:(common_t *)&Net);

    if((void*)stage == &Net || (void*)stage == net_start)
        tab_cnt = 0;
    else
        tab_cnt+=8;

    while(stage != NULL){
        switch(stage->type){
            case NET_ROOT:
            case TLAYER:
            case TSHORTCUT:
                printf("%.*s%.*s%.*s%s\n", tab_cnt, tabulate[0], 4, tabulate[1], ((int)(8-strlen(stage->name))/2 < 0)?0:(int)(8-strlen(stage->name))/2, "    ",  stage->name);
                if(stage->child != NULL)
                    printf_net_structure((common_t *)stage->child);
                break;
            case TCONV:
            case TBACHNORM:
            case TLINER:
                printf("%.*s%.*s%.*s%s\n", tab_cnt, tabulate[0], 4, tabulate[1], ((int)(8-strlen(stage->name))/2 < 0)?0:(int)(8-strlen(stage->name))/2, "    ",  stage->name);
                break;
        default:break;
        }
        stage = (common_t *)stage->sibling;
    }

    if((void*)stage != &Net || (void*)stage == net_start)
        tab_cnt-=8;
}

void recursive_load_net(FILE *file, Tstruct *child, Tstruct* parent){
    data_info_t comn;
    common_t *current;
    int  data_size = 0;
    uint32_t dim[DIM_DEPTH];
    void *data = NULL;
    while(child){
        fseek(file, (int64_t)child, SEEK_SET);
        fread(&comn, sizeof(data_info_t), 1, file);
        data = NULL;
        current = NULL;
        memset(dim, 0x00, sizeof(dim));
        if(comn.type >= TCONV)
            ARRAY_SIZE(comn.dim, data_size);
        switch(comn.type){
            case TLAYER:
            case TSHORTCUT:
                current = (common_t*)node_create((Tstruct*)parent, comn.type, comn.dim, comn.name);
                if(!current){
                    printf("node_create error!\n");
                    return;
                }
                if(comn.reserve != NULL){
                    dbg_print("recursive_load_net\n");
                    recursive_load_net(file, comn.reserve, (Tstruct*)current);
                }
                break;
            case TBACHNORM:
                data_size *= 3;
            case TCONV:
                if(comn.len == BINARY)
                    data_size/=32;
            case TLINER:
                current = (common_t*)node_create((Tstruct*)parent, comn.type, comn.dim, comn.name);
                if(!current){
                    printf("node_create error!\n");
                    return;
                }
                data = ((data_info_t*)current)->data;
                data_size += comn.dim[0];   
                // fseek(file, (long)((data_info_t*)*net)->data, SEEK_SET);
                // fread(data, data_size*sizeof(float), 1, file);
                ((data_info_t*)current)->data = data;
                break;
            default:
                printf("data type error\n");
                break;
        }
        child = comn.sibling;
    }
}

void load_ml_net(char *file_name){
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        eprint("打开文件失败!");
        return;
    }
    dbg_print("open file %s sucess!\n", file_name);
    header_t head;
    fseek(file, 0, SEEK_SET);
    fread(&head, sizeof(header_t), 1, file);
    if(strcmp(head.magic, BNN_MAGIC)){
        printf("file %s is not a binary neural network", file_name);
        fclose(file);
        return;
    }
    dbg_print("file %s header magic correct!\n", file_name);
    net_t **net = &net_start;
    while(*net && (*net)->sibling){
        net = (net_t **)&(*net)->sibling;
        printf("next sibling!\n");
    }
    *net = malloc(sizeof(net_t));
     if (*net == NULL) {
        // 检查 malloc 是否成功
        printf("Memory allocation failed\n");
        return;
    }
    fseek(file, head.node_offset, SEEK_SET);
    fread(*net, sizeof(net_t), 1, file);
    dbg_print("(*net)->name:%s\n", net_start->name);
    Tstruct *child = (*net)->child;
    (*net)->child = NULL;
    if(child)
        recursive_load_net(file, child, (Tstruct*)(*net));

    fclose(file);
    reverse_nodes((common_t *)net_start);
}

int get_child_size(common_t *data){
    if(data == NULL)
        return 0;
    int size_cnt = 0;
    while(data){
        switch(data->type){
            case NET_ROOT:
                size_cnt += alignidx(sizeof(struct net), 4);
                size_cnt += alignidx(get_child_size((common_t *)data->child), 4);
                break;
            case TLAYER:
            case TSHORTCUT:
                size_cnt += alignidx(sizeof(container_t), 4);
                size_cnt += alignidx(get_child_size((common_t *)data->child), 4);
                break;
            case TCONV:
            case TBACHNORM:
            case TLINER:
                size_cnt += alignidx(sizeof(data_info_t), 4);
                break;
            default:break;
        }
        data = (common_t *)data->sibling;
    }
    return size_cnt;
}

void net_data_storage(common_t *data, FILE *file, int node_offset, int *data_offset){
    common_t *stage = (data?data:(common_t *)&Net);
    container_t *cont = malloc(sizeof(container_t));
    data_info_t *d_info = malloc(sizeof(data_info_t));
    int  data_size = 0, sibling_offset = 0;
    
    while(stage != NULL){
        data_size = 0; sibling_offset = 0;
        if(stage->type >= TCONV)
            ARRAY_SIZE(((data_info_t*)stage)->dim, data_size);
        else sibling_offset = get_child_size((common_t *)stage->child);
        // clrprint(32,"node_offset:%d sibling_offset:%d\n", node_offset, sibling_offset);
        switch(stage->type){
            case NET_ROOT:
                goto free_mem;
            case TLAYER:
            case TSHORTCUT:
                fseek(file, node_offset, SEEK_SET);
                memset(cont, 0x00, sizeof(container_t));
                memcpy(cont, stage, sizeof(container_t));
                dbg_print("node_offset:%d %s %s size:%ld\n", node_offset, name_typ[cont->type], cont->name, sizeof(container_t));
                cont->parent = 0;
                node_offset += alignidx(sizeof(container_t), 4);
                if(cont->child)
                    cont->child = (Tstruct *)((int64_t)node_offset);
                if(cont->sibling)
                    cont->sibling = (Tstruct *)((int64_t)node_offset + sibling_offset);
                fwrite(cont, alignidx(sizeof(container_t), 4), 1, file);
                if(stage->child)
                    net_data_storage((common_t *)stage->child, file, node_offset, data_offset);
                node_offset += sibling_offset;
                break;
            case TBACHNORM:
                data_size *= 3;
            case TCONV:
                if(((data_info_t*)stage)->len == BINARY)
                    data_size/=32;
            case TLINER:
                fseek(file, node_offset, SEEK_SET);
                memset(d_info, 0x00, sizeof(data_info_t));
                memcpy(d_info, stage, sizeof(data_info_t));
                dbg_print("node_offset:%d data_offset:0x%x %s %s size:%ld\n", node_offset, *data_offset, name_typ[d_info->type], d_info->name, sizeof(data_info_t));
                node_offset += alignidx(sizeof(data_info_t), 4);
                d_info->parent = 0;
                if(d_info->sibling)
                    d_info->sibling = (Tstruct *)((int64_t)node_offset + sibling_offset);
                d_info->data = (void*)((int64_t)*data_offset);
                fwrite(d_info, alignidx(sizeof(data_info_t), 4), 1, file);

                fseek(file, *data_offset, SEEK_SET);
                data_size += d_info->dim[0];
                dbg_print("data size:%d\n", data_size);
                fwrite(((data_info_t*)stage)->data, data_size*sizeof(float), 1, file);
                *data_offset += data_size*sizeof(float);
                break;
        default:break;
        }
        stage = (common_t *)stage->sibling;
    }
free_mem:
    free(cont);
    free(d_info);
}

void net_storage(char *file_name){
    int node_offset = sizeof(header_t), data_offset = get_child_size((common_t *)&Net)+alignidx(sizeof(header_t), 4);
    header_t head = {
        .magic = BNN_MAGIC,
        .node_offset = node_offset,
        .data_offset = data_offset
    };
    dbg_print("data_offset:%ld", data_offset);
    FILE *file = fopen(file_name, "w");
    if (file == NULL) {
        eprint("打开文件失败!");
        return;
    }
    fseek(file, 0, SEEK_SET);
    fwrite(&head, alignidx(sizeof(header_t), 4), 1, file);

    net_t *net_d = malloc(sizeof(net_t));
    memcpy(net_d, &Net, sizeof(net_t));
    fseek(file, node_offset, SEEK_SET);
    memcpy(net_d->name, file_name, strlen(file_name));
    node_offset += alignidx(sizeof(net_t), 4);
    net_d->child = (Tstruct*)((int64_t)node_offset);
    fwrite(net_d, alignidx(sizeof(net_t), 4), 1, file);

    net_data_storage((common_t *)Net.child, file, node_offset, &data_offset);

    free(net_d);
    fclose(file);
}

void binary_net_data(common_t *data){
    common_t *stage = (data?data:(common_t *)&Net);
    data_info_t *d_info = NULL;
    while(stage != NULL){
        switch(stage->type){
            case NET_ROOT:
            case TLAYER:
            case TSHORTCUT:
                if(stage->child != NULL)
                    binary_net_data((common_t *)stage->child);
                break;
            case TCONV:
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
            break;
        }
        last_stage = current;
        str_ptr = strchr(str_ptr, '.') + 1;
    }
    if(current && current->type >= TCONV){
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
                    int cnt = 0;
                    printf("binary\n");
                    for(uint32_t i = 0;i < array_size/8; i++){
                        printf("%02x ", ((uint8_t *)d_info->data)[i]);
                        if(i%(d_info->dim[1]/8) == (d_info->dim[1]/8)-1)
                            printf(" %d\n",cnt++);
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
                    for(uint32_t i = 0;i < array_size; i++){
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
    else if(current){
        printf("child size:%d\n",get_child_size((common_t*)current->child));
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
    printf_net_structure(NULL);
    binary_net_data(NULL);
    net_storage("resnet18.ml");
    // printf_appoint_data("layer1.");

    // 释放 cJSON 结构体并释放内存
    cJSON_Delete(root);
    free(json_buffer);

    return;
}