#ifndef __UTILS_H__
#define __UTILS_H__
#include "stdio.h"
#include "conv.h"
#include <stdint.h>



void net_lists_add(Net_List *list, void *obj);
void *net_obj_create(Data_Type type);
void BConvTest();
void bitcont_generate();
void Net_Binary(Net_List *Net);
void NetStorage(Net_List* Net, uint64_t *net_cnt, uint64_t *data_cnt, FILE* w, char *file_name);
Net_List *json_model_parse(char* file_name);
Net_List *load_net(char* file_name, uint64_t next, FILE *fp, Layer *last_stage);
void print_net_data(Net_List *Net);

Activate_TypeDef *batchnorm_pcs(Activate_TypeDef *activate, BatchNorm *bn);


void json_model_parse_v2(char* file_name);

#endif
