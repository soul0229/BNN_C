#include <stdint.h>
#include <stdio.h> 
#include <stdlib.h> 
#include <cjson/cJSON.h> 
#include <string.h>
#include <threads.h>
#include "common.h"
#include "core.h"
#include "utils.h"
#include "loadNet.h"

const char *name_order[OMAX] = {[0]=WEIGHT, [1]=BIAS ALPHA, [2]=MEAN, [3]=VAR };
const char *name_typ[] = {
    [TERROR]="error", 
    [NET_ROOT] = "root", 
    [TLAYER] = "layer", 
    [TSHORTCUT] = "shortcut", 
    [TCONV] = "conv", 
    [TBATCHNORM] = "bn",
    [TLINER] = "linear",
    };

const char *tabulate[] = {
    "|       |       |       |       |", 
    "|--------------------------------"
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

static common_t *grep_obj_child(common_t *parent, char *name){
    common_t *ret = parent?(common_t *)parent->child:(common_t *)Net->child;
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
            current = (common_t *)node_create((Tstruct *)last_stage, order, dim, string_buff,0);
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
    common_t *stage = (data?data:(common_t *)Net);
    if(stage->child == NULL && stage->sibling == NULL)
        return;

    if((void*)stage == Net)
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
            case TBATCHNORM:
            case TLINER:
                printf("%.*s%.*s%.*s%s\n", tab_cnt, tabulate[0], 4, tabulate[1], ((int)(8-strlen(stage->name))/2 < 0)?0:(int)(8-strlen(stage->name))/2, "    ",  stage->name);
                break;
        default:
            printf("stage->type error!\n");
            break;
        }
        stage = (common_t *)stage->sibling;
    }

    if((void*)stage != Net)
        tab_cnt-=8;
}

static int get_child_size(common_t *data){
    if(data == NULL)
        return 0;
    int size_cnt = 0;
    while(data){
        switch(data->type){
            case NET_ROOT:
                size_cnt += align16(sizeof(net_storage));
                size_cnt += align16(get_child_size((common_t *)data->child));
                break;
            case TLAYER:
            case TSHORTCUT:
                size_cnt += align16(sizeof(cont_storage));
                size_cnt += align16(get_child_size((common_t *)data->child));
                break;
            case TCONV:
            case TBATCHNORM:
            case TLINER:
                size_cnt += align16(sizeof(data_storage));
                break;
            default:break;
        }
        data = (common_t *)data->sibling;
    }
    return size_cnt;
}

static void net_data_storage(common_t *data, FILE *file, uint32_t node_offset, uint32_t *data_offset){
    common_t *stage = (data?data:(common_t *)Net);
    cont_storage *cont = malloc(sizeof(cont_storage));
    data_storage *d_info = malloc(sizeof(data_storage));
    uint32_t  data_size = 0, sibling_offset = 0;
    
    while(stage != NULL){
        data_size = 0; sibling_offset = 0;
        if(stage->type >= TCONV)
            ARRAY_SIZE(((data_info_t*)stage)->dim, data_size);
        else sibling_offset = get_child_size((common_t *)stage->child);

        switch(stage->type){
            case NET_ROOT:
                goto free_mem;
            case TLAYER:
            case TSHORTCUT:
                fseek(file, node_offset, SEEK_SET);
                memset(cont, 0x00, sizeof(cont_storage));
                cont->type = stage->type;
                strcpy(cont->name, stage->name);
                dbg_print("node_offset:%d %s %s size:%ld\n", node_offset, name_typ[cont->type], cont->name, sizeof(cont_storage));
                node_offset += align16(sizeof(cont_storage));
                if(stage->child)
                    cont->child = node_offset;
                if(stage->sibling)
                    cont->sibling = node_offset + sibling_offset;
                fwrite(cont, sizeof(cont_storage), 1, file);
                if(stage->child)
                    net_data_storage((common_t *)stage->child, file, node_offset, data_offset);
                node_offset += sibling_offset;
                break;
            case TBATCHNORM:
                data_size *= 3;
            case TCONV:
                if(((data_info_t*)stage)->len == BINARY)
                    data_size/=32;
            case TLINER:
                fseek(file, node_offset, SEEK_SET);
                memset(d_info, 0x00, sizeof(data_storage));
                d_info->type = stage->type;
                memcpy(d_info->name, stage->name, sizeof(d_info->name)+sizeof(Data_Len)+sizeof(uint32_t)*DIM_DEPTH);
                dbg_print("node_offset:%d data_offset:0x%x %s %s size:%ld\n", node_offset, *data_offset, name_typ[d_info->type], d_info->name, sizeof(data_storage));
                node_offset += align16(sizeof(data_storage));
                d_info->parent = 0;
                if(stage->sibling)
                    d_info->sibling = node_offset + sibling_offset;
                d_info->data = *data_offset;
                fwrite(d_info, sizeof(data_storage), 1, file);

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

void storage_net(char *file_name){
    if(!Net)
        return;

    uint32_t node_offset = sizeof(header_t), data_offset = get_child_size((common_t *)Net)+align16(sizeof(header_t))+64;
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
    fwrite(&head, sizeof(header_t), 1, file);

    net_storage *net_d = malloc(sizeof(net_storage));
    memset(net_d, 0x00, sizeof(net_storage));
    net_d->type = Net->type;
    fseek(file, node_offset, SEEK_SET);
    strcpy(net_d->name, file_name);
    node_offset += align16(sizeof(net_storage));
    net_d->child = ((uint32_t)node_offset);
    fwrite(net_d, sizeof(net_storage), 1, file);

    net_data_storage((common_t *)Net->child, file, node_offset, &data_offset);

    free(net_d);
    fclose(file);
}

static void binary_net_data(common_t *data){
    common_t *stage = (data?data:(common_t *)Net);
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
                if(d_info->len == FLOAT_BYTE && ((common_t *)d_info->parent)->type != NET_ROOT){
                    uint32_t array_size;
                    ARRAY_SIZE(d_info->dim, array_size);
                    float (*data)[d_info->dim[1]][d_info->dim[2]][d_info->dim[3]] = d_info->data;
                    intx_t (*binary)[d_info->dim[2]][d_info->dim[3]][d_info->dim[1]/DATA_LEN] = malloc((array_size/32 + d_info->dim[0]) * sizeof(float));

                    for(uint16_t dim0 = 0; dim0 < d_info->dim[0]; dim0++){
                        for(uint16_t dim1 = 0; dim1 < d_info->dim[1]; dim1++){
                            for(uint16_t dim2 = 0; dim2 < d_info->dim[2]; dim2++){
                                for(uint16_t dim3 = 0; dim3 < d_info->dim[3]; dim3++){
                                    binary[dim0][dim2][dim3][dim1/DATA_LEN] |= \
                                    ((data[dim0][dim1][dim2][dim3]>0)?((ONE)>>(dim1%DATA_LEN)):((intx_t)(ZERO)));
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
            case TBATCHNORM:
            case TLINER:
                break;
            default:break;
        }
        stage = (common_t *)stage->sibling;
    }
}

void printf_appoint_data(char *str, common_t *data){
    if(str == NULL)
        return;
    int size = 0;
    char string_buff[16],*str_ptr = str;
    common_t *current = NULL, *last_stage = data;
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
                    for(uint32_t i = 0;i < d_info->dim[0]; i++){
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
                    if(offset == 0){
                        for(uint32_t i = 0;i < array_size; i++){
                            printf("%-6f ", data[i]);
                            if(current->type == TCONV){
                                if(i%9 == 8)
                                    printf("\n");
                            }
                            else if(i%16 == 15)
                                printf("\n");
                        }
                    }
                    else {
                        for(uint32_t i = 0;i < d_info->dim[0]; i++){
                            printf("%-6f ", data[i+offset]);
                            if(current->type == TCONV){
                                if(i%9 == 8)
                                    printf("\n");
                            }
                            else if(i%16 == 15)
                                printf("\n");
                        }
                        if(d_info->dim[0]<15)printf("\n");
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
        printf("storage type child size:%d\n",get_child_size((common_t*)current->child));
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
    if(Net == NULL){
        Net = malloc(sizeof(net_t));
        memset(Net, 0x00, sizeof(net_t));
        Net->type = NET_ROOT;
    }
    
    obj_type_detech_create_v2(root);
    reverse_nodes((common_t *)Net);
    printf_net_structure(NULL);
    binary_net_data(NULL);
    storage_net("resnet18.ml");

    // 释放 cJSON 结构体并释放内存
    cJSON_Delete(root);
    free(json_buffer);

    return;
}

void free_net(common_t **net){
    common_t *parent = (common_t *)(*net)->parent;
    common_t *sibling = *net;
    while(sibling){
        switch(sibling->type){
            case NET_ROOT:
                if((common_t *)sibling->child)
                    free_net((common_t **)&sibling->child);
                *net = (common_t *)sibling->sibling;
                dbg_print("%s\n", sibling->name);
                free(sibling);
                break;
            case TLAYER:
            case TSHORTCUT:
                if((common_t *)sibling->child){
                    free_net((common_t **)&sibling->child);
                    ((common_t *)sibling->parent)->child = sibling->sibling;
                }
                dbg_print("%s\n", sibling->name);
                free(sibling);
                break;
            case TCONV:
            case TBATCHNORM:
            case TLINER:
                dbg_print("%s\n", sibling->name);
                ((common_t *)sibling->parent)->child = sibling->sibling;
                free(((data_info_t*)sibling)->data);
                free(sibling);
                break;
            default:break;
        }
        sibling = parent?(common_t *)parent->child:NULL;
    }
}