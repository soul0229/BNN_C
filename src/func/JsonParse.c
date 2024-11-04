#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DEBUG
#include "utils.h"
#include <cjson/cJSON.h>  // cJSON 库的头文件

extern char net_lable[107][64];
const char bn_member[][64] = {"weight", "bias", "running_mean", "running_var"};


Net_List* find_list_header(Net_List* list)
{
    while (list->pre != NULL) {
        list = list->pre;
    }
    return list;
}

Net_List* find_list_tail(Net_List* list)
{
    while (list->next != NULL) {
        list = list->next;
    }
    return list;
}

void net_lists_delete(Net_List* list){
    if(list->pre != NULL)
        list->pre->next = list->next;
    if(list->next != NULL)
        list->next->pre = list->pre;
    free(list);
}

Net_List *get_obj_list(void *obj){
    switch(*((Data_Type *)obj)){
        case    KERNEL:
            return ((CONV_KernelTypeDef *)obj)->list;
        case    BATCHNORM:
            return ((BatchNorm *)obj)->list;
        case    LINEAR:
            return ((Linear *)obj)->list;
        case    LAYER:
            return ((Layer *)obj)->list;
        case    NET_LIST:
            return ((Net_List *)obj);
        default:return NULL;
    }
}

void set_obj_last_stage(void *last_stage, void *obj){
    switch(*((Data_Type *)obj)){
        case    KERNEL:
            ((CONV_KernelTypeDef *)obj)->last_stage = last_stage;
        case    BATCHNORM:
            ((BatchNorm *)obj)->last_stage = last_stage;
        case    LINEAR:
            ((Linear *)obj)->last_stage = last_stage;
        case    LAYER:
        case    NET_LIST:
        default:eprint("set_obj_last_stage: obj type error!\n");
    }
}

void *get_obj_last_stage(void *obj){
    switch(*((Data_Type *)obj)){
        case    KERNEL:
            return ((CONV_KernelTypeDef *)obj)->last_stage;
        case    BATCHNORM:
            return ((BatchNorm *)obj)->last_stage;
        case    LINEAR:
            return ((Linear *)obj)->last_stage;
        case    LAYER:
        case    NET_LIST:
        default:eprint("get_obj_last_stage: obj type error!\n");
            return NULL;
    }
    return NULL;
}

void net_lists_add(Net_List *list, void *obj) {
    list = find_list_tail(list);
    switch(*((Data_Type *)obj)){
        case    KERNEL:
            list->next = ((CONV_KernelTypeDef *)obj)->list;
            ((CONV_KernelTypeDef *)obj)->list->pre = list;
            break;
        case    BATCHNORM:
            list->next = ((BatchNorm *)obj)->list;
            ((BatchNorm *)obj)->list->pre = list;
            break;
        case    LINEAR:
            list->next = ((Linear *)obj)->list;
            ((Linear *)obj)->list->pre = list;
            break;
        case    LAYER:
            list->next = ((Layer *)obj)->list;
            ((Layer *)obj)->list->pre = list;
            break;
        case    NET_LIST:
            list->next = ((Net_List *)obj);
            ((Net_List *)obj)->pre = list;
            break;
        default:break;
    }
}

void net_obj_init(void *site, Data_Type type){
    switch(type){
        case    KERNEL:
            CONV_KernelTypeDef *kernel = site;
            kernel->type = KERNEL;
            kernel->last_stage = NULL;
            kernel->binary = 0;
            kernel->kernel = NULL;
            kernel->list = NULL;
            break;
        case    BATCHNORM:
            BatchNorm *bn = site;
            bn->type = BATCHNORM;
            bn->last_stage = NULL;
            bn->data[MEAN] = NULL;
            bn->data[VAR] = NULL;
            bn->data[WEIGHT] = NULL;
            bn->data[BIAS] = NULL;
            bn->list = NULL;
            break;
        case    LINEAR:
            Linear *linear = site;
            linear->type = LINEAR;
            linear->last_stage = NULL;
            linear->weight = NULL;
            linear->bias = NULL;
            linear->list = NULL;
            break;
        case    LAYER:
            Layer *layer = site;
            layer->type = LAYER;
            layer->data = NULL;
            layer->list = NULL;
            break;
        case    NET_LIST:
            Net_List *list = site;
            list->type = NET_LIST;
            list->data = NULL;
            list->pre = NULL;
            list->next = NULL;
            break;
        default:return;
    }
}

void *net_obj_create(Data_Type type){
    switch(type){
        case    KERNEL:
            CONV_KernelTypeDef *kernel = (CONV_KernelTypeDef *)malloc(sizeof(CONV_KernelTypeDef));
            if(kernel==NULL){
                eprint("net_obj_create kernel error!\n");
                return NULL;
            }
            net_obj_init(kernel, KERNEL);
            kernel->list = net_obj_create(NET_LIST);
            if(kernel->list==NULL){
                free(kernel);
                return NULL;
            }
            kernel->list->data = kernel;
            return kernel;
        case    BATCHNORM:
            BatchNorm *bn = (BatchNorm *)malloc(sizeof(BatchNorm));
            if(bn==NULL){
                eprint("net_obj_create BatchNorm error!\n");
                return NULL;
            }
            net_obj_init(bn, BATCHNORM);
            bn->list = net_obj_create(NET_LIST);
            if(bn->list==NULL){
                free(bn);
                return NULL;
            }
            bn->list->data = bn;
            return bn;
        case    LINEAR:
            Linear *linear = (Linear *)malloc(sizeof(Linear));
            if(linear==NULL){
                eprint("net_obj_create linear error!\n");
                return NULL;
            }
            net_obj_init(linear, LINEAR);
            linear->list = net_obj_create(NET_LIST);
            if(linear->list==NULL){
                free(linear);
                return NULL;
            }
            linear->list->data = linear;
            return linear;
        case    LAYER:
            Layer *layer = (Layer *)malloc(sizeof(Layer));
            if(layer==NULL){
                eprint("net_obj_create layer error!\n");
                return NULL;
            }
            net_obj_init(layer, LAYER);
            layer->list = net_obj_create(NET_LIST);
            if(layer->list==NULL){
                free(layer);
                return NULL;
            }
            layer->list->data = layer;
            return layer;
        case    NET_LIST:
            Net_List *list = (Net_List *)malloc(sizeof(Net_List));
            if(list==NULL){
                eprint("net_obj_create list error!\n");
                return NULL;
            }
            net_obj_init(list, NET_LIST);
            return list;
        default:{
            eprint("net_obj_create error!\n");
            return NULL;
        }
    }
}

void net_obj_delete(void *obj){
    if(obj == NULL)return;
    switch(*((Data_Type *)obj)){
        case    KERNEL:
            CONV_KernelTypeDef *kernel = obj;
            net_lists_delete(kernel->list);
            free(kernel);
            break;
        case    BATCHNORM:
            BatchNorm *bn = obj;
            net_lists_delete(bn->list);
            free(bn);
            break;
        case    LINEAR:
            Linear *linear = obj;
            net_lists_delete(linear->list);
            free(linear);
            break;
        case    LAYER:
            Net_List *list,*list_n;
            list = ((Layer *)obj)->data;
            while(list != NULL && list->data != NULL){
                list_n = list->next;
                net_obj_delete(list->data);
                list = list_n;
            }
            net_lists_delete(((Layer *)obj)->list);
            free(obj);
            break;
        case    NET_LIST:
            net_lists_delete(obj);
            break;
        default:eprint("obj type error!\n");
    }
}

void net_layer_add_data(Layer *layer, void *obj){
    if(layer->data != NULL)
        if(*((Data_Type *)(layer->data)) == NET_LIST){
            Net_List *list = find_list_tail(layer->data);
            net_lists_add(list, obj);
            set_obj_last_stage(layer, obj);
        }
        else {
            eprint("layer->data type error!\n");
        }
    else {
        Net_List *list_t = get_obj_list(obj);
        if(list_t != NULL){
            layer->data = list_t;
            set_obj_last_stage(layer, obj);
        }
        else eprint("obj type error!\n");
    }
}

void read_json_data(cJSON *state_dict, char *lable, float *data){
    cJSON *conv_weight = NULL;
        // 5. 获取 conv1.weight 数组
    conv_weight = cJSON_GetObjectItemCaseSensitive(state_dict, lable);
    if (!cJSON_IsArray(conv_weight)) {
        fprintf(stderr, "Error: %s is not an array\n", lable);
        return;
    }
    else{
        // 6. 遍历数组并访问其中的浮点数值
        for (int i = 0; i < cJSON_GetArraySize(conv_weight); i++) {
            cJSON *item1 = cJSON_GetArrayItem(conv_weight, i);
            if (cJSON_IsArray(item1)) {
                for (int j = 0; j < cJSON_GetArraySize(item1); j++) {
                    cJSON *item2 = cJSON_GetArrayItem(item1, j);
                    if (cJSON_IsArray(item2)) {
                        for (int k = 0; k < cJSON_GetArraySize(item2); k++) {
                            cJSON *item3 = cJSON_GetArrayItem(item2, k);
                            if (cJSON_IsArray(item3)) {
                                for (int l = 0; l < cJSON_GetArraySize(item3); l++) {
                                    cJSON *number = cJSON_GetArrayItem(item3, l);
                                    if (cJSON_IsNumber(number)) {
                                        *(data++) = number->valuedouble;
                                    }
                                }
                            }
                            else{
                                if (cJSON_IsNumber(item3)) {
                                    *(data++) = item3->valuedouble;
                                }
                            }
                        }
                    }
                    else{
                        if (cJSON_IsNumber(item2)) {
                            *(data++) = item2->valuedouble;
                        }
                    }
                }
            }
            else{
                if (cJSON_IsNumber(item1)) {
                    *(data++) = item1->valuedouble;
                }
            }
        }
    }
}

Net_List *obj_type_detech_create(cJSON *state_dict){
    cJSON *member[4] = {NULL};
    void *point = NULL;
    Net_List * Net = NULL;
    uint16_t dim[4] = {1, 1, 1, 1};
    char *pos = NULL, *lable = NULL;
    uint8_t major, minor, conv, bn;
    for(uint8_t cnt; cnt<(sizeof(net_lable)/sizeof(net_lable[0])); cnt++){
        lable = (char*)net_lable[cnt];
        dim[0] = 1;dim[1] = 1;dim[2] = 1;dim[3] = 1;
        member[0] = cJSON_GetObjectItemCaseSensitive(state_dict, lable);
        if(member[0] == NULL)eprint("mem[0]=NULL\n");
        dim[0] = cJSON_GetArraySize(member[0]);
        member[1] = cJSON_GetArrayItem(member[0], 0);
        if (cJSON_IsArray(member[1])) {
            dim[1] = cJSON_GetArraySize(member[1]);
            member[2] = cJSON_GetArrayItem(member[1], 0);
            if (cJSON_IsArray(member[2])) {
                dim[2] = cJSON_GetArraySize(member[2]);
                member[3] = cJSON_GetArrayItem(member[2], 0);
                if (cJSON_IsArray(member[3])) {
                    dim[3] = cJSON_GetArraySize(member[2]);
                }
            }
        }
        dbg_print("dim[%d][%d][%d][%d]\t\t", dim[0], dim[1], dim[2], dim[3]);
        if ((pos = strstr(lable, "layer")) != NULL){
            sscanf(pos, "layer%hhd.%hhd.", &major, &minor);
            dbg_print("layer%d.%d", major, minor);
            if(Net == NULL){
                Net = ((Layer*)net_obj_create(LAYER))->list;
                point = Net->data;
                ((Layer*)point)->major = major;
                ((Layer*)point)->minor = minor;
            }
            else if(*((Data_Type*)(find_list_tail(Net)->data)) != LAYER || \
            ((Layer*)(find_list_tail(Net)->data))->major != major || ((Layer*)(find_list_tail(Net)->data))->minor != minor){
                net_lists_add(find_list_tail(Net), ((Layer*)net_obj_create(LAYER)));
                point = find_list_tail(Net)->data;
                ((Layer*)point)->major = major;
                ((Layer*)point)->minor = minor;
            }
            else {
                point = find_list_tail(Net)->data;
            }
            if((pos = strstr(lable, "conv")) != NULL){
                sscanf(pos, "conv%hhd", &conv);
                dbg_print(".conv%d.weight\n", conv);
                if(((Layer*)point)->data == NULL){
                    ((Layer*)point)->data = ((CONV_KernelTypeDef*)net_obj_create(KERNEL))->list;
                    ((CONV_KernelTypeDef*)(((Net_List*)(((Layer*)point)->data))->data))->last_stage = point;
                    point = ((Net_List*)(((Layer*)point)->data))->data;
                }
                else{
                    net_lists_add(find_list_tail(((Layer*)point)->data), ((CONV_KernelTypeDef*)net_obj_create(KERNEL)));
                    ((CONV_KernelTypeDef*)(find_list_tail(((Layer*)point)->data)->data))->last_stage = point;
                    point = find_list_tail(((Layer*)point)->data)->data;
                }
                    ((CONV_KernelTypeDef*)point)->major = conv;
                    ((CONV_KernelTypeDef*)point)->kernel = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
                    ((CONV_KernelTypeDef*)point)->out_ch = dim[0];
                    ((CONV_KernelTypeDef*)point)->in_ch = dim[1];
                    ((CONV_KernelTypeDef*)point)->size = dim[2];
                    ((CONV_KernelTypeDef*)point)->binary = FLOAT_BYTE;
                    read_json_data(state_dict, lable, ((CONV_KernelTypeDef*)point)->kernel);
            }
            else if((pos = strstr(lable, "bn")) != NULL){
                sscanf(pos, "bn%hhd.", &bn);
                dbg_print(".bn%d", bn);
                if(((Layer*)point)->data == NULL){
                    ((Layer*)point)->data = ((BatchNorm*)net_obj_create(BATCHNORM))->list;
                    ((BatchNorm*)(((Net_List*)(((Layer*)point)->data))->data))->last_stage = point;
                    point = ((Net_List*)(((Layer*)point)->data))->data;
                    ((BatchNorm*)point)->major = bn;
                }else if(*((Data_Type*)(find_list_tail(((Layer*)point)->data)->data)) != BATCHNORM || \
                ((BatchNorm*)(find_list_tail(((Layer*)point)->data)->data))->major != bn ){
                    net_lists_add(find_list_tail(((Layer*)point)->data), ((BatchNorm*)net_obj_create(BATCHNORM)));
                    ((BatchNorm*)(find_list_tail(((Layer*)point)->data)->data))->last_stage = point;
                    point = find_list_tail(((Layer*)point)->data)->data;
                    ((BatchNorm*)point)->major = bn;
                }
                else{
                    point = find_list_tail(((Layer*)point)->data)->data;
                }
                if((pos = strstr(lable, "weight")) != NULL){
                    dbg_print(".weight\n");
                    ((BatchNorm*)point)->size = dim[0]*dim[1]*dim[2]*dim[3];
                    ((BatchNorm*)point)->data[WEIGHT] = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
                    read_json_data(state_dict, lable, ((BatchNorm*)point)->data[WEIGHT]);
                }
                else if((pos = strstr(lable, "bias")) != NULL){
                    dbg_print(".bias\n");
                    ((BatchNorm*)point)->data[BIAS] = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
                    read_json_data(state_dict, lable, ((BatchNorm*)point)->data[BIAS]);
                }
                else if((pos = strstr(lable, "running_mean")) != NULL){
                    dbg_print(".running_mean\n");
                    ((BatchNorm*)point)->data[MEAN] = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
                    read_json_data(state_dict, lable, ((BatchNorm*)point)->data[MEAN]);
                }
                else if((pos = strstr(lable, "running_var")) != NULL){
                    dbg_print(".running_var\n");
                    ((BatchNorm*)point)->data[VAR] = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
                    read_json_data(state_dict, lable, ((BatchNorm*)point)->data[VAR]);
                }
            }
            else if((pos = strstr(lable, "shortcut.0")) != NULL){
                dbg_print(".shortcut.0.weight\n");
                if(((Layer*)point)->data == NULL){
                    ((Layer*)point)->data = ((CONV_KernelTypeDef*)net_obj_create(KERNEL))->list;
                    ((CONV_KernelTypeDef*)(((Net_List*)(((Layer*)point)->data))->data))->last_stage = point;
                    point = ((Net_List*)(((Layer*)point)->data))->data;
                    ((CONV_KernelTypeDef*)point)->major = 3;
                }else {
                    net_lists_add(find_list_tail(((Layer*)point)->data), ((CONV_KernelTypeDef*)net_obj_create(KERNEL)));
                    ((CONV_KernelTypeDef*)(find_list_tail(((Layer*)point)->data)->data))->last_stage = point;
                    point = find_list_tail(((Layer*)point)->data)->data;
                    ((CONV_KernelTypeDef*)point)->major = 3;
                }
                ((CONV_KernelTypeDef*)point)->kernel = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
                ((CONV_KernelTypeDef*)point)->out_ch = dim[0];
                ((CONV_KernelTypeDef*)point)->in_ch = dim[1];
                ((CONV_KernelTypeDef*)point)->size = dim[2];
                ((CONV_KernelTypeDef*)point)->binary = FLOAT_BYTE;
                read_json_data(state_dict, lable, ((CONV_KernelTypeDef*)point)->kernel);
            }
            else if((pos = strstr(lable, "shortcut.1")) != NULL){
                dbg_print(".shortcut.1");
                if(((Layer*)point)->data == NULL){
                    ((Layer*)point)->data = ((BatchNorm*)net_obj_create(BATCHNORM))->list;
                    ((BatchNorm*)(((Net_List*)(((Layer*)point)->data))->data))->last_stage = point;
                    point = ((Net_List*)(((Layer*)point)->data))->data;
                    ((BatchNorm*)point)->major = 3;
                }else if(*((Data_Type*)(find_list_tail(((Layer*)point)->data)->data)) != BATCHNORM || \
                ((BatchNorm*)(find_list_tail(((Layer*)point)->data)->data))->major != 3 ){
                    net_lists_add(find_list_tail(((Layer*)point)->data), ((BatchNorm*)net_obj_create(BATCHNORM)));
                    ((BatchNorm*)(find_list_tail(((Layer*)point)->data)->data))->last_stage = point;
                    point = find_list_tail(((Layer*)point)->data)->data;
                    ((BatchNorm*)point)->major = 3;
                }
                else{
                    point = find_list_tail(((Layer*)point)->data)->data;
                }
                if((pos = strstr(lable, "weight")) != NULL){
                    dbg_print(".weight\n");
                    ((BatchNorm*)point)->size = dim[0]*dim[1]*dim[2]*dim[3];
                    ((BatchNorm*)point)->data[WEIGHT] = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
                    read_json_data(state_dict, lable, ((BatchNorm*)point)->data[WEIGHT]);
                }
                else if((pos = strstr(lable, "bias")) != NULL){
                    dbg_print(".bias\n");
                    ((BatchNorm*)point)->data[BIAS] = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
                    read_json_data(state_dict, lable, ((BatchNorm*)point)->data[BIAS]);
                }
                else if((pos = strstr(lable, "running_mean")) != NULL){
                    dbg_print(".running_mean\n");
                    ((BatchNorm*)point)->data[MEAN] = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
                    read_json_data(state_dict, lable, ((BatchNorm*)point)->data[MEAN]);
                }
                else if((pos = strstr(lable, "running_var")) != NULL){
                    dbg_print(".running_var\n");
                    ((BatchNorm*)point)->data[VAR] = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
                    read_json_data(state_dict, lable, ((BatchNorm*)point)->data[VAR]);
                }
            }
        }else if((pos = strstr(lable, "bn")) != NULL){
            sscanf(pos, "bn%hhd.", &bn);
            dbg_print("bn%d", bn);
            if(Net == NULL){
                Net = ((BatchNorm*)net_obj_create(BATCHNORM))->list;
                point = Net->data;
                ((BatchNorm*)point)->major = bn;
            }else if(*((Data_Type*)(find_list_tail(Net)->data)) != BATCHNORM || \
                ((BatchNorm*)(find_list_tail(Net)->data))->major != bn ){
                net_lists_add(find_list_tail(Net), ((BatchNorm*)net_obj_create(BATCHNORM)));
                point = find_list_tail(Net)->data;
                ((BatchNorm*)point)->major = bn;
            }
            else {
                point = find_list_tail(Net)->data;
            }
            if((pos = strstr(lable, "weight")) != NULL){
                dbg_print(".weight\n");
                ((BatchNorm*)point)->size = dim[0]*dim[1]*dim[2]*dim[3];
                ((BatchNorm*)point)->data[WEIGHT] = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
                read_json_data(state_dict, lable, ((BatchNorm*)point)->data[WEIGHT]);
            }
            else if((pos = strstr(lable, "bias")) != NULL){
                dbg_print(".bias\n");
                ((BatchNorm*)point)->data[BIAS] = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
                read_json_data(state_dict, lable, ((BatchNorm*)point)->data[BIAS]);
            }
            else if((pos = strstr(lable, "running_mean")) != NULL){
                dbg_print(".running_mean\n");
                ((BatchNorm*)point)->data[MEAN] = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
                read_json_data(state_dict, lable, ((BatchNorm*)point)->data[MEAN]);
            }
            else if((pos = strstr(lable, "running_var")) != NULL){
                dbg_print(".running_var\n");
                ((BatchNorm*)point)->data[VAR] = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
                read_json_data(state_dict, lable, ((BatchNorm*)point)->data[VAR]);
            }
        }
        else if((pos = strstr(lable, "conv")) != NULL){
            sscanf(pos, "conv%hhd", &conv);
            dbg_print("conv%d.weight\n", conv);
            if(Net == NULL){
                Net = ((CONV_KernelTypeDef*)net_obj_create(KERNEL))->list;
                point = Net->data;
            }
            else{
                net_lists_add(find_list_tail(Net), ((CONV_KernelTypeDef*)net_obj_create(KERNEL)));
                point = find_list_tail(Net)->data;
            }
            ((CONV_KernelTypeDef*)point)->major = conv;
            ((CONV_KernelTypeDef*)point)->kernel = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
            ((CONV_KernelTypeDef*)point)->out_ch = dim[0];
            ((CONV_KernelTypeDef*)point)->in_ch = dim[1];
            ((CONV_KernelTypeDef*)point)->size = dim[2];
            ((CONV_KernelTypeDef*)point)->binary = FLOAT_BYTE;
            read_json_data(state_dict, lable, ((CONV_KernelTypeDef*)point)->kernel);
        }
        else if((pos = strstr(lable, "linear")) != NULL){
            dbg_print("linear");
            if(Net == NULL){
                Net = ((Linear*)net_obj_create(LINEAR))->list;
                point = Net->data;
                ((Linear*)point)->major = 1;
                ((Linear*)point)->size[0] = dim[0];
                ((Linear*)point)->size[1] = dim[1];
            }else if(*((Data_Type*)(find_list_tail(Net)->data)) != LINEAR || \
                ((Linear*)(find_list_tail(Net)->data))->major != 1 ){
                net_lists_add(find_list_tail(Net), ((Linear*)net_obj_create(LINEAR)));
                point = find_list_tail(Net)->data;
                ((Linear*)point)->major = 1;
                ((Linear*)point)->size[0] = dim[0];
                ((Linear*)point)->size[1] = dim[1];
            }
            else{
                point = find_list_tail(Net)->data;
            }
            if((pos = strstr(lable, "weight")) != NULL){
                dbg_print(".weight\n");
                ((Linear*)point)->weight = (float *)calloc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(float));
                read_json_data(state_dict, lable, ((Linear*)point)->weight);
            }
            else if((pos = strstr(lable, "bias")) != NULL){
                dbg_print(".bias\n");
                ((Linear*)point)->bias = (float *)calloc(dim[0], sizeof(float));
                read_json_data(state_dict, lable, ((Linear*)point)->bias);
            }
        }
    }
    return Net;
}

Net_List *json_model_parse(char* file_name) {
    // 打开并读取 model.json 文件
    FILE *fp = fopen(file_name, "r");
    if (fp == NULL) {
        perror("Error opening file");
        return NULL;
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
        return NULL;
    }

    // 获取 state_dict 对象
    cJSON *state_dict = cJSON_GetObjectItemCaseSensitive(root, "state_dict");
    if (!cJSON_IsObject(state_dict)) {
        fprintf(stderr, "Error: state_dict is not an object\n");
        cJSON_Delete(root);
        return NULL;
    }
    Net_List *Net = obj_type_detech_create(state_dict);
    Net_Binary(Net);

    // 释放 cJSON 结构体并释放内存
    cJSON_Delete(root);
    free(json_buffer);

    return Net;
}

