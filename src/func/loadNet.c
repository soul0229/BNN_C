#include "loadNet.h"
#include <stdint.h>
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include "common.h"
#include "core.h"
#include "utils.h"

static void recursive_load_net(FILE *file, uint32_t child, Tstruct* parent){
    data_storage comn;
    common_t *current;
    uint32_t data_size = 0, total_size = 0;
    uint32_t dim[DIM_DEPTH];
    void *data = NULL;
    while(child){
        fseek(file, child, SEEK_SET);
        fread(&comn, sizeof(data_storage), 1, file);
        data = NULL;
        total_size = 0;
        current = NULL;
        memset(dim, 0x00, sizeof(dim));
        if(comn.type >= TCONV)
            ARRAY_SIZE(comn.dim, data_size);
        switch(comn.type){
            case TLAYER:
            case TSHORTCUT:
                current = (common_t*)node_create((Tstruct*)parent, comn.type, comn.dim, comn.name,0);
                if(!current){
                    printf("node_create error!\n");
                    return;
                }
                if(comn.reserve != 0){
                    dbg_print("recursive_load_net\n");
                    recursive_load_net(file, comn.reserve, (Tstruct*)current);
                }
                break;
            case TBATCHNORM:
                total_size = data_size * 2;
            case TCONV:
                
            case TLINER:
                total_size += data_size;
                if(comn.len == BINARY)
                    total_size = data_size/32;
                current = (common_t*)node_create((Tstruct*)parent, comn.type, comn.dim, comn.name, comn.len);
                if(!current){
                    printf("node_create error!\n");
                    return;
                }
                total_size = (total_size+comn.dim[0])*sizeof(float);   
                fseek(file, (long)(comn.data), SEEK_SET);
                fread(((data_info_t*)current)->data, total_size, 1, file);
                break;
            default:
                printf("data type error\n");
                break;
        }
        child = comn.sibling;
    }
}

void load_ml_net(char *file_name){
    if(Net == NULL){
        Net = malloc(sizeof(net_t));
        memset(Net, 0x00, sizeof(net_t));
        Net->type = NET_ROOT;
    }

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
        printf("file %s is not a binary neural network\n", file_name);
        fclose(file);
        return;
    }
    dbg_print("file %s header magic correct!\n", file_name);
    net_t **net = &Net;
    while(*net && (*net)->sibling){
        net = (net_t **)&(*net)->sibling;
        printf("next sibling!\n");
    }

    net_storage temp;
    *net = malloc(sizeof(net_t));
     if (*net == NULL) {
        // 检查 malloc 是否成功
        printf("Memory allocation failed\n");
        return;
    }
    fseek(file, head.node_offset, SEEK_SET);
    fread(&temp, sizeof(net_storage), 1, file);
    dbg_print("(*net)->name:%s\n", Net->name);
    (*net)->type = temp.type;
    strcpy((*net)->name, temp.name);
    if(temp.child)
        recursive_load_net(file, temp.child, (Tstruct*)(*net));

    fclose(file);
    reverse_nodes((common_t *)*net);
}